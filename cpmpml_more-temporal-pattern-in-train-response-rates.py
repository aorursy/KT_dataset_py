

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df = pd.read_csv( "../input/train.csv")



df["qmax"]      = df.apply( lambda row: max(row["qid1"], row["qid2"]), axis=1 )

df              = df.sort_values(by=["qmax"], ascending=True)

df["dupe_rate"] = df.is_duplicate.rolling(window=500, min_periods=500).mean()

df["timeline"]  = np.arange(df.shape[0]) / float(df.shape[0])



df.plot(x="timeline", y="dupe_rate", kind="line")

plt.show()
df["qdiff"]      = df.apply( lambda row: abs(row["qid1"] - row["qid2"]), axis=1 )

df              = df.sort_values(by=["qdiff"], ascending=True)

df["dupe_rate_diff"] = df.is_duplicate.rolling(window=500, min_periods=500).mean()

df["timeline"]  = np.arange(df.shape[0]) / float(df.shape[0])



df.plot(x="timeline", y="dupe_rate_diff", kind="line")

plt.show()