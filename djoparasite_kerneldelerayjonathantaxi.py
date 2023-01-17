import numpy as np

import pandas as pd 

import os

import datetime

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit

print(os.listdir("../input"))
def haversine_array(lat1, lng1, lat2, lng2): 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    AVG_EARTH_RADIUS = 6371 # in km 

    lat = lat2 - lat1 

    lng = lng2 - lng1 

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2 

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d)) 

    return h



def bearing_array(lat1, lng1, lat2, lng2): 

    AVG_EARTH_RADIUS = 6371 # in km 

    lng_delta_rad = np.radians(lng2 - lng1) 

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2)) 

    y = np.sin(lng_delta_rad) * np.cos(lat2) 

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad) 

    return np.degrees(np.arctan2(y, x))
def preprocess(df):

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    df['pickup_year'] = df['pickup_datetime'].dt.year

    df['pickup_month'] = df['pickup_datetime'].dt.month

    df['pickup_day'] = df['pickup_datetime'].dt.day

    df['pickup_hour'] = df['pickup_datetime'].dt.hour + 1

    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday + 1

    df['pickup_minute'] = df['pickup_datetime'].dt.minute

    df['pickup_seconde'] = df['pickup_datetime'].dt.minute * 60

    df['store_and_fwd_flag'] = pd.get_dummies(df['store_and_fwd_flag'], drop_first=True)

    df['pickup_datetime'] = pd.to_numeric(df['pickup_datetime'], errors='coerce')

    

    df['bearing'] = df.apply(lambda x: bearing_array(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']),axis=1)

    df.loc[:, 'center_latitude'] = (df['pickup_latitude'].values + df['dropoff_latitude'].values) / 2 

    df.loc[:, 'center_longitude'] = (df['pickup_longitude'].values + df['dropoff_longitude'].values) / 2

    df['distance'] = df.apply(lambda x: haversine_array(x['pickup_latitude'], x['pickup_longitude'], x['dropoff_latitude'], x['dropoff_longitude']),axis=1)

    

    
def get_columns_selected(df, excludes):

    preprocess(df)

    columns = df.columns.tolist()

    return [c for c in columns if c not in excludes]
dataDir = '../input/'

df_train = pd.read_csv(dataDir + 'train.csv', index_col='id')

df_test = pd.read_csv(dataDir + 'test.csv', index_col='id')
df_train = df_train[df_train.trip_duration < 3600]
EXCLUDES = ['trip_duration', 'dropoff_datetime']

df_train['passenger_count'] = df_train.passenger_count.map(lambda x: 1 if x == 0 else x)

df_train = df_train[df_train.passenger_count <= 6]

X = df_train[get_columns_selected(df_train, EXCLUDES)]

y = df_train.trip_duration

X.shape, y.shape
cv = ShuffleSplit(4, test_size=0.01, train_size=0.02, random_state=0)

rf = RandomForestRegressor()

losses = -cross_val_score(rf, X, y, cv=cv, scoring='neg_mean_squared_log_error')

losses = [np.sqrt(l) for l in losses]

np.mean(losses)
rf.fit(X, y)
df_test.head()
X_test = df_test[get_columns_selected(df_test, [])]

y_pred = rf.predict(X_test)

y_pred.mean()
submission = pd.read_csv(dataDir + 'sample_submission.csv') 

submission.head()
submission['trip_duration'] = y_pred

submission.head()
submission.describe()
submission.to_csv('submission.csv', index=False)