import collections

import pandas as pd

import lightgbm as lgb
dd = "../input/lish-moa"

# read data

ss = pd.read_csv(f"{dd}/sample_submission.csv", index_col=0)

df_test = pd.read_csv(f"{dd}/test_features.csv", index_col=0)

df_train = pd.read_csv(f"{dd}/train_features.csv", index_col=0)

df_train_sco = pd.read_csv(f"{dd}/train_targets_scored.csv", index_col=0)

df_train_nsco = pd.read_csv(f"{dd}/train_targets_nonscored.csv", index_col=0)
df_train.drop(["cp_time", "cp_type", "cp_dose"], axis=1, inplace=True)

df_test.drop(["cp_time", "cp_type", "cp_dose"], axis=1, inplace=True)
for col in df_train_sco.columns:

    clf = lgb.LGBMClassifier(random_state=0)

    clf.fit(df_train, df_train_sco[col])

    probas = clf.predict_proba(df_test)[:, 1]

    ss[col] = probas

ss.to_csv("submission.csv")