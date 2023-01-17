import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
train_0 = train.query("target == 0")

train_0_shrink = train_0.head(int(train_0.shape[0] * 0.12))

y_0 = train_0_shrink['target']

X_0 = train_0_shrink.drop('target', axis=1)

X_0 = X_0.drop('ID_code', axis=1)

ID_0 = train_0_shrink['ID_code']
train_1 = train.query("target == 1")

y_1 = train_1['target']

X_1 = train_1.drop('target', axis=1)

X_1 = X_1.drop('ID_code', axis=1)

ID_1 = train_1['ID_code']
X = pd.concat([X_0, X_1])

y = pd.concat([y_0, y_1])

ID_Code = pd.concat([ID_0, ID_1])
y.sum() / len(y)
from sklearn.ensemble import RandomForestClassifier

# ベンチマーク

rfc = RandomForestClassifier(random_state=0, n_estimators=100)

rfc.fit(X, y)
ID_test = test['ID_code']

X_test = test.drop('ID_code', axis=1)
y_test = rfc.predict(X_test)
submit = pd.read_csv("../input/sample_submission.csv")

submit["target"] = y_test

submit.to_csv("submission.csv", index=False)