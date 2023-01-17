# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df.head()
df = df[(df['passenger_count'] >= 1)]

df = df[(df['trip_duration'] <= 7000)]
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['year'] = df['pickup_datetime'].dt.year

df['day'] = df['pickup_datetime'].dt.day

df['month'] = df['pickup_datetime'].dt.month

df['hour'] = df['pickup_datetime'].dt.hour

df['minute']= df['pickup_datetime'].dt.minute

df['second']= df['pickup_datetime'].dt.second



df.head()
df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

df_test['year'] = df_test['pickup_datetime'].dt.year

df_test['day'] = df_test['pickup_datetime'].dt.day

df_test['month'] = df_test['pickup_datetime'].dt.month

df_test['hour'] = df_test['pickup_datetime'].dt.hour

df_test['minute']= df_test['pickup_datetime'].dt.minute

df_test['second']= df_test['pickup_datetime'].dt.second



df_test.head()
selected_columns = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','year','day','month','hour','minute','second']

X = df[selected_columns]

y = df['trip_duration']

X.shape, y.shape
X_test = df_test[selected_columns]

rf = RandomForestRegressor()

rs = ShuffleSplit(n_splits=3, test_size=.12, train_size=.25)

# Cross Validation

score = -cross_val_score(rf, X, y, cv=rs, scoring='neg_mean_squared_log_error')

score.mean()
rf.fit(X, y)
y_pred = rf.predict(X_test)

y_pred.mean()
df_submission = pd.read_csv('../input/sample_submission.csv')

df_submission.head()
df_submission['trip_duration'] = y_pred

df_submission.head()
df_submission.describe()
df_submission.to_csv('submission.csv', index=False)