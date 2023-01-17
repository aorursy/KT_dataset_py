import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
df_train = pd.read_csv('../input/train.csv', index_col='id')

df_test = pd.read_csv('../input/test.csv', index_col='id')
df_train.head()
df_train.info()
df_train.describe()
# for key in df_train:

#     n_keys = len(df_train[key].unique())

#     print(f'key: {key}')

#     print(f'unique keys number: {n_keys}')

#     print(df_train[key].unique())
# unique, counts = np.unique(df_train.dropoff_datetime, return_counts=True)

# dict(zip(unique, counts))
# u, c = np.unique(counts, return_counts=True)

# dict(zip(u, c))
# df_train['times'] = df_train.pickup_datetime.apply(str) + ' ' +  df_train.dropoff_datetime.apply(str)

# _unique, _counts = np.unique(df_train.times, return_counts=True)

# dict(zip(_unique, _counts))
# _u, _c = np.unique(_counts, return_counts=True)

# dict(zip(_u, _c))
# empty_courses = df_train[df_train['passenger_count'] == 0]

# empty_courses
from geopy.distance import geodesic

from haversine import haversine
# coords = tuple(lat, lon)

# def calculate_distance(coords1, coords2):

#     return geodesic(coords1, coords2).miles # in miles



def calculate_distance(coords1, coords2):

    return haversine(coords1, coords2) # in kilometers
# from math import sin, cos, sqrt, atan2, radians



# # function taken on stackoverflow

# def calculate_distance(longitude1, latitude1, longitude2, latitude2):

#     # approximate radius of earth in km

#     R = 6373.0



#     lat1 = radians(latitude1)

#     lon1 = radians(longitude1)

#     lat2 = radians(latitude2)

#     lon2 = radians(longitude2)



#     dlon = lon2 - lon1

#     dlat = lat2 - lat1



#     a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

#     c = 2 * atan2(sqrt(a), sqrt(1 - a))

#     # return distance in km

#     return R * c



def add_distance(df):

    return df.apply(lambda row: calculate_distance(

        (row.pickup_latitude, row.pickup_longitude),

        (row.dropoff_latitude, row.dropoff_longitude)

    ), axis=1)



df_train['distance'] = add_distance(df_train)



df_test['distance'] = add_distance(df_test)



df_train['distance'].head()
periods = {

    'morning': {

        'min': 6,

        'max': 12

    },

    'afternoon': {

        'min': 12,

        'max': 18

    },

    'evening': {

        'min': 18,

        'max': 24

    },

    'night': {

        'min': 0,

        'max': 6

    },

}

rush_hours = [

    (7, 9),

    (16, 18)

]



def get_time_period(hour):

    for p in periods:

        if hour >= periods[p]['min'] and hour < periods[p]['max']:

            return p



def is_during_rush(hour):

    for h in rush_hours:

        if hour >= h[0] and hour < h[1]:

            return True

        return False

from datetime import datetime



def extract_date_features(df, col):

    """ Extract features from a date. """

    df[col + '_year'] = df[col].dt.year

    df[col + '_month'] = df[col].dt.month

    df[col + '_week'] = df[col].dt.week

    df[col + '_day'] = df[col].dt.day

    df[col + '_dow'] = df[col].dt.dayofweek

    df[col + '_hour'] = df[col].dt.hour

    df[col + '_minute'] = df[col].dt.minute

    df[col + '_weekday'] = df[col].dt.weekday

    df[col + '_days_in_month'] = df[col].dt.days_in_month

    df[col + '_is_month_start'] = df[col].dt.is_month_start

    df[col + '_is_month_end'] = df[col].dt.is_month_end

    df[col + '_period'] = df[col].dt.hour.apply(get_time_period)

    df['is_during_rush_hour'] = df[col].dt.hour.apply(is_during_rush)

    return df



df_train['pickup_datetime_dt'] = df_train.pickup_datetime.apply(lambda row: datetime.strptime(row, '%Y-%m-%d %H:%M:%S'))

# df_train['dropoff_datetime_dt'] = df_train.dropoff_datetime.apply(lambda row: datetime.strptime(row, '%Y-%m-%d %H:%M:%S'))

df_test['pickup_datetime_dt'] = df_test.pickup_datetime.apply(lambda row: datetime.strptime(row, '%Y-%m-%d %H:%M:%S'))

df_train = extract_date_features(df_train, 'pickup_datetime_dt')

df_test = extract_date_features(df_test, 'pickup_datetime_dt')
# df_meteo = pd.read_csv('./weather_nyc_2016.csv')

# df_meteo.head()
# meteo_filtered = df_meteo.loc[:, ['Time','Temp.','Conditions']]

# df_meteo['date'] = df_meteo['Time'].apply(lambda row: datetime.strptime(row, '%Y-%m-%d %H:%M:%S'))

# meteo_filtered['year'] = df_meteo['date'].dt.year

# meteo_filtered['pickup_datetime_dt_month'] = df_meteo['date'].dt.month

# meteo_filtered['pickup_datetime_dt_day'] = df_meteo['date'].dt.day

# meteo_filtered['pickup_datetime_dt_hour'] = df_meteo['date'].dt.hour

# meteo_filtered = meteo_filtered[meteo_filtered['year'] == 2016]



# meteo_filtered.head()
# train = pd.merge(df_train, meteo_filtered[['Temp.', 'Conditions','pickup_datetime_dt_month', 'pickup_datetime_dt_day', 'pickup_datetime_dt_hour']], on=['pickup_datetime_dt_month', 'pickup_datetime_dt_day', 'pickup_datetime_dt_hour'], how='left')

# test = pd.merge(df_test, meteo_filtered[['Temp.', 'Conditions','pickup_datetime_dt_month', 'pickup_datetime_dt_day', 'pickup_datetime_dt_hour']], on=['pickup_datetime_dt_month', 'pickup_datetime_dt_day', 'pickup_datetime_dt_hour'], how='left')

train = df_train

test = df_test
train.head()
train.store_and_fwd_flag = [1 if flag == 'N' else 2 for flag in train.store_and_fwd_flag]
test.store_and_fwd_flag = [1 if flag == 'N' else 2 for flag in test.store_and_fwd_flag]
def to_category(keys):

    for key in keys:

        train[key] = train[key].astype('category')

        test[key] = test[key].astype('category')



CATEGORIES = [

    'vendor_id',

    'store_and_fwd_flag',

    'passenger_count'

]

to_category(CATEGORIES)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
# train
rf = RandomForestRegressor(n_estimators=10)



COLUMNS = [

    'pickup_datetime_dt_month',

    'pickup_datetime_dt_day',

    'pickup_datetime_dt_dow',

    'pickup_datetime_dt_hour',

    'pickup_longitude',

    'pickup_latitude',

    'dropoff_longitude',

    'dropoff_latitude',

    'distance',

    'passenger_count',

    'vendor_id'

]



X_train = train[COLUMNS]

y_train = train.trip_duration

X_test = test[COLUMNS]
from math import log, exp
# new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(X_train, y_train, random_state=42)

# new_y_train = [log(y) for y in new_y_train]

# rf.fit(new_X_train, new_y_train)

# new_y_hat = rf.predict(new_X_test)

# new_y_hat = [exp(y_hat) for y_hat in new_y_hat]
# test_df_with_y_hat = new_X_test

# test_df_with_y_hat['pred'] = new_y_hat

# # test_df_with_y_hat
# def calcul_error(y_hat, y):

#     return abs(y.trip_duration - y_hat.pred) / y.trip_duration



# def calcul_all_error(df_y_hat, df_y):

#     res = []

#     for index in df_y_hat.index:

#         # print(index)

#         res.append(calcul_error(df_y_hat.loc[index], df_y.loc[index]))

#     return res



# list_error = calcul_all_error(test_df_with_y_hat, train)

# np.mean(list_error)*100
# X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_train, y_train, test_size=0.90, random_state=42)

# len(X_temp_train)
# from sklearn.model_selection import cross_val_score

# rf_cv = RandomForestRegressor(n_estimators=10)

# cv_losses = -cross_val_score(rf_cv, X_temp_train, y_temp_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_log_error')
# cv_losses
# np.mean(cv_losses), np.std(cv_losses)
rf_X_train = X_train

rf_y_train = y_train

rf_y_train = [log(y) for y in rf_y_train]

rf.fit(rf_X_train, rf_y_train)

y_hat = rf.predict(X_test)
y_hat = [exp(y) for y in y_hat]
np.mean(y_hat)
sample = pd.read_csv('../input/sample_submission.csv')
sample.trip_duration = y_hat

sample.to_csv('results.csv', index=False)