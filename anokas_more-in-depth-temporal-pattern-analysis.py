

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df = pd.read_csv( "../input/train.csv")



df["qmax"]      = df.apply( lambda row: max(row["qid1"], row["qid2"]), axis=1 )

df              = df.sort_values(by=["qmax"], ascending=True)

df["dupe_rate"] = df.is_duplicate.rolling(window=500, min_periods=500).mean()

df["timeline"]  = np.arange(df.shape[0]) / float(df.shape[0])



plt.figure(figsize=(20, 20))

#df.plot(x="timeline", y="dupe_rate", kind="line")

plt.plot(df['timeline'],)

plt.show()
df["dupe_rate_2"] = df.is_duplicate.rolling(window=25000, min_periods=500).mean()

plt.figure(figsize=(20, 20))

df.plot(x="timeline", y="dupe_rate_2", kind="line")

plt.show()
plt.hist(df['dupe_rate_2'].dropna(), bins=50)
df['dupe_rate_grad'] = df['dupe_rate_2'].diff(5000)

df.plot(x="timeline", y="dupe_rate_grad", kind="line")