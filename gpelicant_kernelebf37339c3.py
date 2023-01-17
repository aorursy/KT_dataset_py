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
df = pd.read_csv("../input/train.csv")

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df = df[df['trip_duration'] <= 5000]

df.info()
## split pickup_datetime

df['pickup_hour'] = df['pickup_datetime'].dt.hour

df['pickup_minute'] = df['pickup_datetime'].dt.minute

df['pickup_second'] = df['pickup_datetime'].dt.second
df.head()
## X creation

X = df[[

    'pickup_longitude', 

    'pickup_latitude', 

    'dropoff_longitude', 

    'dropoff_latitude', 

    'pickup_hour',

    'pickup_minute',

    'pickup_second',

   ]]
## Y creation

Y = df[['trip_duration']]



## Shape

X.shape, Y.shape
## Cross validation

rf = RandomForestRegressor()

score = -cross_val_score(rf, X, Y, cv=5, scoring='neg_mean_squared_log_error')

score.mean()
rf.fit(X, Y)
## DF Test creation

df_test = pd.read_csv('../input/test.csv')

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

df_test.head()
## split pickup_datetime

df_test['pickup_hour'] = df_test['pickup_datetime'].dt.hour

df_test['pickup_minute'] = df_test['pickup_datetime'].dt.minute

df_test['pickup_second'] = df_test['pickup_datetime'].dt.second
## X creation

X_test = df_test[[

    'pickup_longitude', 

    'pickup_latitude', 

    'dropoff_longitude', 

    'dropoff_latitude', 

    'pickup_hour',

    'pickup_minute',

    'pickup_second',

    ]]
## y prediction and mean for this prediction

y_pred = rf.predict(X_test)

y_pred.mean()
## submit creation

submission = pd.read_csv('../input/sample_submission.csv') 

submission.head()
## submission for the y prediction

submission['trip_duration'] = y_pred

submission.head()
## describe submission

submission.describe()
## send submission

submission.to_csv('submission.csv', index=False)
!ls
