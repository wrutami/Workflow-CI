"""Re-training model RandomForest untuk dataset Bank Marketing (dijalankan lewat `mlflow run`).

Tracking memakai penyimpanan lokal (./mlruns) karena GitHub Actions tidak dapat mengakses
MLflow Tracking UI lokal. Parameter diterima dari entry point MLProject.
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=16)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--data_dir", default="bank_marketing_preprocessing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train.drop(columns=["y"]), train["y"]
    X_test, y_test = test.drop(columns=["y"]), test["y"]

    mlflow.sklearn.autolog(log_input_examples=True)
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
            max_features="log2", class_weight="balanced", random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_roc_auc": roc_auc_score(y_test, y_proba),
            "test_pr_auc": average_precision_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)
        print({k: round(v, 4) for k, v in metrics.items()})

        # Simpan run_id agar workflow CI bisa mengambil model dari run ini.
        Path("last_run_id.txt").write_text(run.info.run_id)
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()
