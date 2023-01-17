import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import lightgbm as lgb

import os
train_data = pd.read_csv('../input/liverpool-ion-switching/train.csv')

test_data = pd.read_csv('../input/liverpool-ion-switching/test.csv')

submission = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')
test_data.head()
train_data.head()
train_data.tail()
Factors = ["time", "signal"]

x = train_data[Factors]

y = train_data.open_channels

x_train, x_valid, y_train, y_valid = train_test_split(x,y,train_size=0.3,test_size=0.5,random_state=0)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1, n_jobs=-1)

model.fit(x_train, y_train)
model1 = LinearDiscriminantAnalysis()

model1.fit(x_train, y_train)
test_cols = ["time", "signal"]

test_pred = test_data[test_cols]

pred = model.predict(test_pred)

pred1 = model1.predict(test_pred)
score = model.score(x_valid, y_valid)

score1 = model1.score(x_valid, y_valid)

print(score)

print(score1)
submission = pd.DataFrame()

submission['time'] = test_data['time']

submission['open_channels'] = pred1

submission.to_csv('submission.csv', index=False, float_format='%.4f')
submission.head()