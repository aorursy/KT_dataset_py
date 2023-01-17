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



X_all = np.concatenate([X_train, X_test])

X_all = (X_all - X_all.mean()) / X_all.std()



X_train = X_all[:X_train.shape[0]]

X_test = X_all[X_train.shape[0]:]
# parameter search

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier

lcv = LogisticRegressionCV(penalty="l1",

                           Cs=100,

                           cv=50,

                           solver="liblinear",

                           verbose=50)

lcv.fit(X_train, y)
# coef path

path = lcv.coefs_paths_[1].mean(axis=0)

for i in range(301):

    plt.plot(path[:, i], lw=1, alpha=.3)
# cv score

plt.plot(lcv.scores_[1].mean(axis=0))
plt.barh(np.arange(300), lcv.coef_[0])
# prediction

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold

skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=2434)

y_oof = np.zeros(y.shape)

y_pred = np.zeros(X_test.shape[0])



for dev, val in tqdm(skf.split(X_train, y), total=100):

    X_dev = X_train[dev]

    y_dev = y[dev]

    X_val = X_train[val]

    lr = LogisticRegression(penalty="l1", C=lcv.C_[0], solver="liblinear")

    lr.fit(X_dev, y_dev)

    y_oof[val] += lr.predict_proba(X_val)[:, 1] / 10

    y_pred += lr.predict_proba(X_test)[:, 1] / 100
# cv score(auc)

from sklearn.metrics import roc_auc_score

roc_auc_score(y, y_oof)
# submit prediction

smpsb_df["target"] = y_pred

smpsb_df.to_csv("simple_lasso.csv", index=None)