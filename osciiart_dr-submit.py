import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics
df_sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")

df_tmp = pd.read_csv("../input/dr-submission/df_test_DR_Ensemble_ver5.csv")

df_sub = pd.merge(df_sub[['id_code']], df_tmp[['id_code', 'pred_c']], on='id_code', how='left').fillna(0)

df_sub.columns = ['id_code', 'diagnosis']

df_sub["diagnosis"] = df_sub["diagnosis"].astype(int)

df_sub.to_csv("submission.csv", index=None)

df_sub.head(30)