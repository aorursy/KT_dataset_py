import pandas as pd

import seaborn as sns

import pathlib as Path

import matplotlib.pyplot as plt

import sklearn



import numpy as np

from sklearn.model_selection import ShuffleSplit

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score





import os

print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv')

df_train.head()

df_train = df_train[df_train['passenger_count'] >= 1] 

df_train = df_train[df_train['trip_duration'] <= 5000]
plt.figure(figsize=(20, 5))

sns.set(style="darkgrid")

sns.countplot(x="trip_duration", data=df_train);
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])

df_train['year'] = df_train['pickup_datetime'].dt.year

df_train['month'] = df_train['pickup_datetime'].dt.month

df_train['day'] = df_train['pickup_datetime'].dt.day

df_train['hour'] = df_train['pickup_datetime'].dt.hour

df_train['minute'] = df_train['pickup_datetime'].dt.minute

df_train['second'] = df_train['pickup_datetime'].dt.second
df_train.describe()
selected_columns = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',

                  'dropoff_latitude','year','month','day','hour','minute',

                  'second']
X_train = df_train[selected_columns]

y_train = df_train['trip_duration']

rf = RandomForestRegressor()

random_split = ShuffleSplit(n_splits=3, test_size=0.05, train_size=0.1, random_state=0)

looses = -cross_val_score(rf, X_train, y_train, cv=random_split, scoring='neg_mean_squared_log_error')

looses = [np.sqrt(l) for l in looses]

np.mean(looses)
rf.fit(X_train, y_train)
df_test = pd.read_csv('../input/test.csv')

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

df_test.head()

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

df_test['year'] = df_test['pickup_datetime'].dt.year

df_test['month'] = df_test['pickup_datetime'].dt.month

df_test['day'] = df_test['pickup_datetime'].dt.day

df_test['hour'] = df_test['pickup_datetime'].dt.hour

df_test['minute'] = df_test['pickup_datetime'].dt.minute

df_test['second'] = df_test['pickup_datetime'].dt.second
X_test = df_test[selected_columns]
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv('../input/sample_submission.csv')

submission['trip_duration'] = y_pred

submission.to_csv('submission.csv', index=False)