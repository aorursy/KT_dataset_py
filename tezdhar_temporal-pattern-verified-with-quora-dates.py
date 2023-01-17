

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df = pd.read_csv( "../input/train.csv")



df_new      = pd.concat([df[['qid1', 'question1','is_duplicate']].rename(columns = {'qid1': 'qid','question1': 'question'}), 

                         df[['qid2', 'question2','is_duplicate']].rename(columns = {'qid2': 'qid','question2': 'question'})])

df_new      = df_new.sort_values(by=["qid"], ascending=True)

df_new["dupe_rate"] = df_new.is_duplicate.rolling(window=500, min_periods=500).mean()

df_new["timeline"]  = np.arange(df_new.shape[0]) / float(df_new.shape[0])



df_new.plot(x="timeline", y="dupe_rate", kind="line")

plt.show()
df_new2 = df_new.drop_duplicates(subset='qid')

df_new2["dupe_rate"] = df_new2.is_duplicate.rolling(window=500, min_periods=500).mean()

df_new2["timeline"]  = np.arange(df_new2.shape[0]) / float(df_new2.shape[0])

df_new2.plot(x="timeline", y="dupe_rate", kind="line")

plt.show()
df_new2.question.iloc[3]
df_new2.question.iloc[10000]
df_new2.question.iloc[200000]
df_new2.question.iloc[300000]
df_new2.question.iloc[-50000]
df_new2.question.iloc[-3]
df_new2.shape