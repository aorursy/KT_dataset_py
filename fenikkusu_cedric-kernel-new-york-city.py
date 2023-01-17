import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from os import path

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

from math import acos, cos, radians, sin

print(os.listdir("../input"))
PATH = "../input"

df = pd.read_csv(PATH + '/train.csv')

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df.head()
df.describe()
df.info()
def distance(latitude1, longitude1, latitude2, longitude2):

    return (6366 * acos(

    cos(radians(latitude1)) *

    cos(radians(latitude2)) *

    cos(radians(longitude2) - radians(longitude1)) +

    sin(radians(latitude1)) *

    sin(radians(latitude2))   

    ))
pickup_latitude = df['pickup_latitude']

pickup_longitude = df['pickup_longitude']

dropoff_latitude = df['dropoff_latitude']

dropoff_longitude = df['dropoff_longitude']

test = []



for la1, lo1, la2, lo2 in zip(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):

    test = distance(la1, lo1, la2, lo2)
df['year_pickup_datetime'] = df['pickup_datetime'].dt.year

df['month_pickup_datetime'] = df['pickup_datetime'].dt.month

df['day_pickup_datetime'] = df['pickup_datetime'].dt.day

df['hour_pickup_datetime'] = df['pickup_datetime'].dt.hour

df['minute_pickup_datetime'] = df['pickup_datetime'].dt.minute

df['seconde_pickup_datetime'] = df['pickup_datetime'].dt.minute * 60

df['weekday_pickup_datetime'] = df['pickup_datetime'].dt.weekday

df['distance'] = test



df_filter = (df['passenger_count'] > 0) & (df['trip_duration'] > 120) & (df['trip_duration'] < 10800)

df_result = df.loc[df_filter]

df_result.shape
columns = ['passenger_count', 'vendor_id', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'year_pickup_datetime', 'month_pickup_datetime', 

          'day_pickup_datetime', 'hour_pickup_datetime', 'minute_pickup_datetime', 'seconde_pickup_datetime', 'weekday_pickup_datetime','distance']

X = df_result[columns]

y = df_result['trip_duration']

X.shape, y.shape
rf = RandomForestRegressor()

rs = ShuffleSplit(n_splits=3, test_size=.7, random_state=42)

#rs.get_n_splits(rf, X, y)

score = -cross_val_score(rf, X, y, cv=rs, scoring='neg_mean_squared_log_error')

np.sqrt(score.mean())
rf.fit(X, y)
df_test = pd.read_csv(PATH + '/test.csv')

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])

df_test.head()
colomns_test = ['passenger_count', 'vendor_id', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'year_pickup_datetime', 'month_pickup_datetime', 

          'day_pickup_datetime', 'hour_pickup_datetime', 'minute_pickup_datetime', 'seconde_pickup_datetime', 'weekday_pickup_datetime','distance']

df_test['year_pickup_datetime'] = df_test['pickup_datetime'].dt.year

df_test['month_pickup_datetime'] = df_test['pickup_datetime'].dt.month

df_test['day_pickup_datetime'] = df_test['pickup_datetime'].dt.day

df_test['hour_pickup_datetime'] = df_test['pickup_datetime'].dt.hour

df_test['minute_pickup_datetime'] = df_test['pickup_datetime'].dt.minute

df_test['seconde_pickup_datetime'] = df_test['pickup_datetime'].dt.minute * 60

df_test['weekday_pickup_datetime'] = df_test['pickup_datetime'].dt.weekday

df_test['distance'] = df['distance']

df_test.head()
X_test = df_test[colomns_test]
y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv(PATH + '/sample_submission.csv')

submission.head()
submission['trip_duration'] = y_pred

submission.head()
submission.describe()
submission.to_csv('result.csv', index=False)