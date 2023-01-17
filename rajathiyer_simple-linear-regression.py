import pandas as pd

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt

from sklearn.metrics import log_loss

from sklearn.linear_model import LinearRegression
train = pd.read_csv("../input/lish-moa/train_features.csv")

test = pd.read_csv("../input/lish-moa/test_features.csv")

score = pd.read_csv("../input/lish-moa/train_targets_scored.csv")

train.head()
test.head()
score.head()
print("Training missing data:", train.isnull().sum().sum())

print("Testing missing data:", test.isnull().sum().sum())

print("Target missing data:", score.isnull().sum().sum())

X_train   = train.drop(['sig_id'],axis=1)

y_train   = score.drop(['sig_id'],axis=1)

X_test    = test.drop(['sig_id'],axis=1)
X_train       = pd.get_dummies(X_train)

X_test        = pd.get_dummies(X_test)
regressor  = LinearRegression()

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_train)

log_loss(y_train,predictions)
predictions = regressor.predict(X_test)

sample = pd.read_csv('../input/lish-moa/sample_submission.csv')

sample.iloc[:,1:] = predictions

sample.to_csv('submission.csv',index=False)
sample.head()