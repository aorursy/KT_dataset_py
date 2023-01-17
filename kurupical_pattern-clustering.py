# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# read csv
df = pd.read_csv("../input/application_train.csv")
# FLAG_DOCUMENT_cluster
cols = ["FLAG_DOCUMENT_{}".format(i) for i in range(2, 22)]
df_cluster = df[cols].groupby(cols).size().reset_index()
df_cluster.columns = cols + ["size"]
df_cluster = df_cluster.sort_values("size", ascending=False).head(10)
# show pattern of FLAG_DOCUMENTS
df_cluster["size"]
# show pattern of FLAG_DOCUMENTS(percentage)
df_cluster["size"] / len(df)
# clustering record
df_doc_flg = df[["SK_ID_CURR"] + cols]
df["doc_cluster"] = [0] * len(df)
for i in range(len(df_cluster)):
    df["doc_cluster"] += np.all(df[cols].values == df_cluster[cols].iloc[i].values, axis=1).astype(int) * (i + 1)
# calculate average target of all cluster
df_output = None
col = "doc_cluster"
for target in [0, 1]:
    w_df = df.query("TARGET == {}".format(target)).groupby([col]).size().reset_index()
    size_name = "target_{}_count".format(target)
    w_df.columns = ["cluster", size_name]
    if df_output is None:
        df_output = w_df
    else:
        df_output = pd.merge(df_output, w_df, how="outer", on=["cluster"])
df_output["target_1_ratio"] = df_output["target_1_count"] / df_output[["target_0_count", "target_1_count"]].sum(axis=1)
df_output
print(df["TARGET"].mean())
# see cluster 1
df_cluster.iloc[0]