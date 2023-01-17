import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')

ss = pd.read_csv('../input/sample_submission.csv')

test = pd.read_csv('../input/test.csv')

train.head()
train.shape
train['target'].value_counts()
X_train = train.iloc[:, 2:]

y_train = train['target']

X_test = test.iloc[:, 1:]
from xgboost.sklearn import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
ss.head()
ss['target'] = y_pred
ss.to_csv('prediction.csv', index = False)