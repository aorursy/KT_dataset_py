import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_train_sample = pd.read_csv("../input/train_sample.csv")
print("Training Samples contains {} rows {} columns".format(*df_train_sample.shape))
df_train_sample.head()
# It's extremely imbalanced
df_train_sample.groupby("is_attributed").agg({"ip": "count"})
def get_precision(x):
    return x.sum() / x.shape[0]
def get_precision_recall_by_single_feature(col, df):
    df_precision_recall = pd.DataFrame(columns=["Count", "Precision", "Recall"], index=df[col].unique())
    for c, f in [("Precision", get_precision), ("Count", "count"), ("Recall", "sum")]:
        _df = df.groupby(col).agg({"is_attributed": f})
        df_precision_recall.loc[_df.index, c] = _df["is_attributed"]
    df_precision_recall["Recall"] = df_precision_recall["Recall"] / df["is_attributed"].sum()
    return df_precision_recall
col = "app"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())
col = "device"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())
col = "os"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())
col = "channel"
df_app_precision_recall = get_precision_recall_by_single_feature(col, df_train_sample)
# Get top 5 most indicative app
display(df_app_precision_recall.sort_values("Precision", ascending=False).head())
# Get top 5 most download app
display(df_app_precision_recall.sort_values("Recall", ascending=False).head())
