import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization library



%matplotlib inline
TRAIN_PATH = os.path.join("..", "input", "train.csv")

TEST_PATH = os.path.join("..", "input", "test.csv")



train = pd.read_csv(TRAIN_PATH)

test = pd.read_csv(TEST_PATH)
train = train[train['passenger_count']>= 1]
train = train[train['trip_duration']>= 90 ]

train = train[train['trip_duration']<= 10800 ]
train = train.loc[train['pickup_longitude']> -80]
train = train.loc[train['pickup_latitude']< 44]
train = train.loc[train['dropoff_longitude']> -90]
train = train.loc[train['dropoff_latitude']> 34]
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
train['hour'] = train.loc[:,'pickup_datetime'].dt.hour;

train['week'] = train.loc[:,'pickup_datetime'].dt.week;

train['weekday'] = train.loc[:,'pickup_datetime'].dt.weekday;

train['hour'] = train.loc[:,'pickup_datetime'].dt.hour;

train['month'] = train.loc[:,'pickup_datetime'].dt.month;
test['hour'] = test.loc[:,'pickup_datetime'].dt.hour;

test['week'] = test.loc[:,'pickup_datetime'].dt.week;

test['weekday'] = test.loc[:,'pickup_datetime'].dt.weekday;

test['hour'] = test.loc[:,'pickup_datetime'].dt.hour;

test['month'] = test.loc[:,'pickup_datetime'].dt.month;
train['dist'] = (abs(train['pickup_latitude']-train['dropoff_latitude'])

                        + abs(train['pickup_longitude']-train['dropoff_longitude']))

test['dist'] = (abs(test['pickup_latitude']-test['dropoff_latitude'])

                        + abs(test['pickup_longitude']-test['dropoff_longitude']))
y_train = train["trip_duration"]

X_train = train[["vendor_id", "passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude", "dist", "hour", "week", "weekday", "month" ]]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42)
m = RandomForestRegressor(n_estimators=20,min_samples_leaf=100, min_samples_split=150)

m.fit(X_train, y_train)
X_test = test[["vendor_id", "passenger_count","pickup_longitude", "pickup_latitude","dropoff_longitude","dropoff_latitude","dist", "hour", "week", "weekday", "month"]]

prediction = m.predict(X_test)

prediction
my_submission = pd.DataFrame({'id': test.id, 'trip_duration': prediction})

my_submission.head()
my_submission.to_csv('submission.csv', index=False)