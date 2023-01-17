import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np





from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/train.csv')

tst = pd.read_csv('../input/test.csv')

df.info()
df.head()
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['year'] = df['pickup_datetime'].dt.year

df['month'] = df['pickup_datetime'].dt.month

df['day'] = df['pickup_datetime'].dt.day

df['weekday'] = df['pickup_datetime'].dt.weekday

df['hour'] = df['pickup_datetime'].dt.hour
df = df[df['trip_duration']<= 50000]

df = df[df['passenger_count']>= 1]
df.describe()
selected_columns = ['passenger_count',

                    'pickup_latitude',

                    'pickup_longitude',

                    'dropoff_latitude',

                    'dropoff_longitude',

                    'year', 'month', 

                    'day', 'hour',

                    'weekday'

                   ]
tst['pickup_datetime'] = pd.to_datetime(tst['pickup_datetime'])

tst['year'] = tst['pickup_datetime'].dt.year

tst['month'] = tst['pickup_datetime'].dt.month

tst['day'] = tst['pickup_datetime'].dt.day

tst['weekday'] = tst['pickup_datetime'].dt.weekday

tst['hour'] = tst['pickup_datetime'].dt.hour
tst.describe()
X = df[selected_columns]

y = df['trip_duration']
X.shape, y.shape
rf = RandomForestRegressor()

cv = ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.1, random_state=200)

losses = -cross_val_score(rf, X_train, y_train, cv=cv, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(l) for l in losses]

losses
rf.fit(X, y)
tst = pd.read_csv('../input/test.csv')

tst['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

tst.head()
X_test = tst[selected_columns]
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv("../input/sample_submission.csv")

submission.head()
submission.to_csv("submission.csv", index=False)
!ls
pd.read_csv("submission.csv")