import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df['datetime'] = pd.to_datetime(df['datetime'])

df.info()
df['year'] = df['datetime'].dt.year

df['month'] = df['datetime'].dt.month

df['day'] = df['datetime'].dt.day

df['weekday'] = df['datetime'].dt.weekday

df['hour'] = df['datetime'].dt.hour

df.describe()

selected_columns = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'year', 'month', 

                    'day', 'hour', 'weekday']

X = df[selected_columns]

y = df['count']

X.shape, y.shape
rf = RandomForestRegressor()

score = -cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_log_error')

score.mean()
rf.fit(X, y)
df_test = pd.read_csv('../input/test.csv')

df_test['datetime'] = pd.to_datetime(df_test['datetime'])

df_test.head()
df_test['year'] = df_test['datetime'].dt.year

df_test['month'] = df_test['datetime'].dt.month

df_test['day'] = df_test['datetime'].dt.day

df_test['weekday'] = df_test['datetime'].dt.weekday

df_test['hour'] = df_test['datetime'].dt.hour
X_test = df_test[selected_columns]
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv('../input/sampleSubmission.csv') 

submission.head()
submission['count'] = y_pred

submission.head()
submission.describe()
submission.to_csv('submission.csv', index=False)
!ls
