import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from tqdm import tqdm



import os

print(os.listdir("../input"))
# load data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

smpsb_df = pd.read_csv("../input/sample_submission.csv")
# normalization

X_train = train_df.iloc[:, 2:].values

X_test = test_df.iloc[:, 1:].values

y = train_df["target"].values
# prediction

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, random_state=42)

y_oof = np.ones(y.shape) * 0.5

y_pred = np.ones(X_test.shape[0]) * 0.5



for dev, val in tqdm(skf.split(X_train, y), total=10):

    X_dev = X_train[dev]

    y_dev = y[dev]

    X_val = X_train[val]

    lr = LogisticRegression(penalty="l1", C=1, solver="liblinear")

    lr.fit(X_dev, y_dev)

    y_oof[val] = lr.predict_proba(X_val)[:, 1]

    y_pred += lr.predict_proba(X_test)[:, 1] / 10
# cv score(auc)

from sklearn.metrics import roc_auc_score

roc_auc_score(y, y_oof)
# submit prediction

smpsb_df["target"] = y_pred

smpsb_df.to_csv("simple_logreg_l1.csv", index=None)