# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/nyc-taxi-trip-duration"))



# Any results you write to the current directory are saved as output.
# Import libraries

import datetime

import math



import geopy.distance



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



%matplotlib inline
train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',

                   parse_dates=['pickup_datetime', 'dropoff_datetime'],

                   dtype={'store_and_fwd_flag':'category'})
train.info()
train.head()
train.shape
train.describe(include='all')
test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',

                   parse_dates=['pickup_datetime'],

                   dtype={'store_and_fwd_flag':'category'})
test.describe(include='all')
def create_datetime_columns(df, col_name, scaler=None):

    '''

    Create addtional datetime columns and scale them

    '''



    raw_data = pd.concat([df[col_name].dt.dayofweek,

                      df[col_name].dt.month,

                      df[col_name].dt.week,

                      df[col_name].dt.hour], axis=1)

    

    if not scaler:

        scaler = MinMaxScaler()

        scaler.fit(raw_data)

    

    scaled_data = scaler.transform(raw_data)



    df[col_name+ '_' + 'dayofweek'] = scaled_data[:, 0]

    df[col_name+ '_' + 'month'] = scaled_data[:, 1]

    df[col_name+ '_' + 'week'] = scaled_data[:, 2]

    df[col_name+ '_' + 'hour'] = scaled_data[:, 3]

        

    return df, scaler
def get_distance_km(row):

    '''

    Get a distance in kilometers between a pickup and dropoff locations of a given rows

    '''

    coords_1 = (row.pickup_latitude, row.pickup_longitude)

    coords_2 = (row.dropoff_latitude, row.dropoff_longitude)

    

    return geopy.distance.geodesic(coords_1, coords_2).km
def get_distance_pickup_to_timesquare_km(row):

    

    coords_timesquare = (40.7590, -73.9845)

    

    coords_pickup = (row.pickup_latitude, row.pickup_longitude)

    

    return geopy.distance.geodesic(coords_pickup, coords_timesquare).km
def get_distance_dropoff_to_timesquare_km(row):

    

    coords_timesquare = (40.7590, -73.9845)

    

    coords_dropoff = (row.dropoff_latitude, row.dropoff_longitude)

    

    return geopy.distance.geodesic(coords_dropoff, coords_timesquare).km
def transform_data(df, scalers=None, cleanData=False):

    

    if scalers:

        scaler_datetime = scalers['datetime']

        scaler_distance = scalers['distance']

        scaler_passenger_count = scalers['passenger_count']

    else:

        scaler_datetime = None

        scaler_distance = None

        scaler_passenger_count = None

    

    data_clean = df.copy()

    

    #### Categorical column (store_and_fwd_flag)

    # This column must be converted to a numerical value by 

    # using cat.codes and cast it to int

    data_clean = pd.concat([data_clean.drop('vendor_id', axis=1), 

                        pd.get_dummies(data_clean['vendor_id'], prefix='vendor_id')], axis=1)

    

    data_clean = pd.concat([data_clean.drop('store_and_fwd_flag', axis=1), 

                        pd.get_dummies(data_clean['store_and_fwd_flag'], prefix='store_and_fwd_flag')], axis=1)

    

    

    #### Datetime columns (pickup_datetime)

    # datetime columns which is **pickup_datetime**

    # should be split to 'dayofweek', 'dayofyear', 'weekofyear', 'month', 'hour'

#     data_clean = create_datetime_columns(data_clean, 

#                                          ['pickup_datetime', 'dropoff_datetime'])

    # Only do get additional column for pickup_datetime should be enought because

    # They are typically on the same day

    data_clean, scaler_datetime = create_datetime_columns(data_clean, 'pickup_datetime', scaler=scaler_datetime)



    #### Location columns (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)

    # Create a new column **distance_km** to store a distance value in km computed from (pickup_longitude	pickup_latitude	dropoff_longitude	dropoff_latitude)

    data_clean['distance_km'] = data_clean.apply(lambda row: get_distance_km(row), axis=1)

    data_clean['dist_pickup_to_timesquare'] = data_clean.apply(lambda row: get_distance_pickup_to_timesquare_km(row), axis=1)

    data_clean['dist_dropoff_to_timesquare'] = data_clean.apply(lambda row: get_distance_dropoff_to_timesquare_km(row), axis=1)



    raw_distance = pd.concat([data_clean['distance_km'],

                          data_clean['dist_pickup_to_timesquare'], 

                          data_clean['dist_dropoff_to_timesquare']], axis=1)

    

    if not scaler_distance:

        scaler_distance = StandardScaler()

        scaler_distance.fit(raw_distance)

    

    scaled_distance = scaler_distance.transform(raw_distance)

    

    data_clean['scaled_distance_km'] = scaled_distance[:, 0]

    data_clean['scaled_dist_pickup_to_timesquare'] = scaled_distance[:, 1]

    data_clean['scaled_dist_dropoff_to_timesquare'] = scaled_distance[:, 2]

    

    if cleanData:

        # After doing the exploratory analysis, I found that there are outliers in the dataset

        # (there are trips that have 1k km) that could potentially cause an unexpected behavior

        # Hence, remove those outlier data before proceeding         

        data_clean = data_clean[data_clean.distance_km < data_clean.distance_km.quantile(0.99)]

    

    

    #### Passenger count

    # Apply MinMaxScaler to the **passenger_count**

    data_passenger_count = np.array(data_clean['passenger_count']).reshape(-1, 1)

    

    if not scaler_passenger_count:

        scaler_passenger_count = MinMaxScaler()

        scaler_passenger_count.fit(data_passenger_count)



    scaled_passenger_count = scaler_passenger_count.transform(data_passenger_count)

    data_clean['scaled_passenger_count'] = scaled_passenger_count[:,0]

    

    #### Drop unused columns

    # **id** column can be dropped because we do not need it in training

    # **pickup_datetime** and **dropoff_datetime** must be dropped after all above are done



    data_clean = data_clean.drop(['id', 

                                  'pickup_datetime',

                                  'distance_km',

                                  'dist_pickup_to_timesquare',

                                  'dist_dropoff_to_timesquare',

                                  'passenger_count'

                                 ], axis=1)

    

    # Test data does not have dropof_datetime column. Hence, skip it

    if data_clean.columns.contains('dropoff_datetime'):

        data_clean = data_clean.drop(['dropoff_datetime'], axis=1)

        

    

    out_scalers = {'datetime': scaler_datetime, 

                   'distance': scaler_distance,

                   'passenger_count': scaler_passenger_count}

    

    return data_clean, out_scalers
# # Clean the train data

# # Comment it out and use the saved clean data instead if it is already created



# print('[{}] Start'.format(datetime.datetime.now()))



# %time data_clean, out_scalers = transform_data(train, cleanData=True)



# data_clean.reset_index().to_feather('data_clean')
# # We will use a saved clean data from the previous session here

data_clean = pd.read_feather('../input/nyctaxi-clean-train-data/data_clean')
data_clean.head()
# Inspect the output dataframe

data_clean.sample(20)
data_clean.info()
corr = data_clean.corr()
plt.figure(figsize=(8,6))



sns.heatmap(corr);
corr.style.background_gradient(cmap='coolwarm')
# Get all column names

data_clean.columns.tolist()
X = data_clean.drop(['trip_duration'], axis=1)

y = data_clean['trip_duration']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                    test_size=0.2, 

                                                    random_state=42)
X_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m, X_train, X_valid, y_train, y_valid):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor()



print('[{}] Start'.format(datetime.datetime.now()))



%time m.fit(X_train, y_train)

print('[{}] Start'.format(datetime.datetime.now()))





%time print_score(m, X_train, X_valid, y_train, y_valid)
y_pred_train = m.predict(X_train)
y_pred = m.predict(X_valid)
# From https://stackoverflow.com/questions/46202223/root-mean-log-squared-error-issue-with-scitkit-learn-ensemble-gradientboostingre

def rmsle(y, y0):

    return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
rmsle(y_train, y_pred_train)
rmsle(y_valid, y_pred)
test.head()
test.shape
# # Clean the test data

# # Comment it out and use the saved clean data instead if it is already created



# print('[{}] Start'.format(datetime.datetime.now()))



# %time test_clean, _ = transform_data(test, scalers=out_scalers)



# test_clean.reset_index().to_feather('test_clean')
# # We will use a saved clean data from the previous session here

test_clean = pd.read_feather('../input/nyctaxi-clean-test-data/test_clean')
test_clean.head()
test_clean.info()
test_clean.shape
X_sub = test_clean.copy()
y_sub = m.predict(X_sub)
df_sub = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')
df_sub.head()
df_sub['trip_duration'] = y_sub

df_sub.head()
df_sub.shape
df_sub.to_csv('submission_default_scaling.csv', index=False)