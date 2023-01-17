import os

import warnings



from IPython.display import display

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd









warnings.filterwarnings('ignore')

%matplotlib inline
print(os.listdir("../nyc-taxi-trip-duration"))
train = pd.read_csv("../input/train.csv", index_col="id")

test = pd.read_csv("../input/test.csv", index_col="id")
train.head(15)
len(train.index) == train.index.nunique()
train.info()
train.describe()
index = train.index

index
columns = train.columns

columns
values = train.values

values
train['vendor_id'] = train.vendor_id.astype('category')

train['passenger_count'] = train.passenger_count.astype('category')

train['store_and_fwd_flag'] = train.store_and_fwd_flag.astype('category')
train.info()
train['passenger_count'].value_counts(normalize=True).plot(kind="barh");
train['vendor_id'].value_counts(normalize=True).plot(kind="barh");
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

#train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
def extract_date_features(train, col):

   X = pd.DataFrame()

   """ Extract features from a date. """

   X['year'] = train[col].dt.year

   X['month'] = train[col].dt.month

   X['day'] = train[col].dt.day

   X['week_of_year'] = train[col].dt.week

   X['day_of_week'] = train[col].dt.dayofweek

   X['hour'] = train[col].dt.hour

   X['minute'] = train[col].dt.minute

   X['second'] = train[col].dt.second

   return X
pu_dt = extract_date_features(train, 'pickup_datetime')

pu_dt.head()
#do_dt = extract_date_features(train, 'dropoff_datetime')

#do_dt.head()
pu_dt.shape
train_2 = pd.concat([train, pu_dt], axis=1)

train_2.info()
#weather = pd.read_csv("weather_nyc_2016.csv", parse_dates=['Time'])

#weather.head(5)
#weather.info()
#weather_filtered = weather.iloc[:, [0,1]]

#weather_filtered['year'] = weather_filtered['Time'].dt.year

#weather_filtered['month'] = weather_filtered['Time'].dt.month

#weather_filtered['day'] = weather_filtered['Time'].dt.day

#weather_filtered['hour'] = weather_filtered['Time'].dt.hour

#weather_filtered = weather_filtered[weather_filtered['year'] == 2016]

#weather_filtered.head(5)
#train_3 = pd.merge(train_2, weather_filtered[['Temp.', 'month', 'day', 'hour']], on = ['month', 'day', 'hour'], how = 'left')
train_2.head(10)
from math import sin, cos, sqrt, atan2, radians



def calculate_distance(longitude1, latitude1, longitude2, latitude2):

    # approximate radius of earth in km

    R = 6373.0



    lat1 = radians(latitude1)

    lon1 = radians(longitude1)

    lat2 = radians(latitude2)

    lon2 = radians(longitude2)



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # return distance in km

    return R * c



def add_distance(df):

    return df.apply(lambda row: calculate_distance(

        row.pickup_longitude,

        row.pickup_latitude,

        row.dropoff_longitude,

        row.dropoff_latitude

    ), axis=1)



train_2['distance'] = add_distance(train_2)



#df_test['distance'] = add_distance(df_test)
train_2.head(15)
train_2.info()
y_train = train_2['trip_duration']

y_train.shape
X_train = train_2.loc[:, train_2.columns != 'trip_duration']

X_train.shape
train_2.info()
SELECTED_COLUMNS = [

    'year',

    'month',

    'day',

    'day_of_week',

    'hour',

    'minute',

    'second',

    'distance'

]
f_X_train = train_2[SELECTED_COLUMNS]
f_X_train.info()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
rf = RandomForestRegressor()

rf
sub_X_train, sub_X_test, sub_y_train, sub_y_test = train_test_split(f_X_train, y_train)
cv_losses = -cross_val_score(rf, sub_X_train, sub_y_train, cv=5, n_jobs=-1, scoring='neg_mean_squared_log_error')
cv_losses
rf.fit(f_X_train, y_train)
np.mean(cv_losses), np.std(cv_losses)
#f_X_train
#sub_X_train, sub_X_test, sub_y_train, sub_y_test = train_test_split(X_train, y_train)
test.head(15)
len(train.index) == train.index.nunique()
test.describe()
test['vendor_id'] = test.vendor_id.astype('category')

test['passenger_count'] = test.passenger_count.astype('category')
test.info()
test['passenger_count'].value_counts(normalize=True).plot(kind="barh");
test['vendor_id'].value_counts(normalize=True).plot(kind="barh");
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
pu_dt_test = extract_date_features(test, 'pickup_datetime')

pu_dt_test.head()
test_2 = pd.concat([test, pu_dt_test], axis=1)

test_2.info()
test_2.head(5)
#weather.head(5)
#test_3 = pd.merge(test_2, weather_filtered[['Temp.', 'Conditions','month', 'day', 'hour']], on = ['month', 'day', 'hour'], how = 'left')
#test_3.head(5)
from math import sin, cos, sqrt, atan2, radians



def calculate_distance(longitude1, latitude1, longitude2, latitude2):

    # approximate radius of earth in km

    R = 6373.0



    lat1 = radians(latitude1)

    lon1 = radians(longitude1)

    lat2 = radians(latitude2)

    lon2 = radians(longitude2)



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # return distance in km

    return R * c



def add_distance(df):

    return df.apply(lambda row: calculate_distance(

        row.pickup_longitude,

        row.pickup_latitude,

        row.dropoff_longitude,

        row.dropoff_latitude

    ), axis=1)



test_2['distance'] = add_distance(test_2)



#df_test['distance'] = add_distance(df_test)
f_X_test = test_2[SELECTED_COLUMNS]
f_X_test.head(5)
y_hat = rf.predict(f_X_test)
np.mean(y_hat)
submission = pd.read_csv('../input/sample_submission.csv')

submission.trip_duration = y_hat

submission.to_csv('results.csv', index=False)