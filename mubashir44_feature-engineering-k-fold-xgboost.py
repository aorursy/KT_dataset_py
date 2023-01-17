# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import timedelta

import datetime as dt

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans



# Any results you write to the current directory are saved as output.
t0 = dt.datetime.now()

train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv')

test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')

sample_submission = pd.read_csv('../input/nyc-taxi-trip-duration/sample_submission.csv')

test_1 = test.copy()
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday

train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear

train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour

train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']



test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday

test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear

test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']



train.loc[:, 'pickup_dayofyear'] = train['pickup_datetime'].dt.dayofyear

test.loc[:,'pickup_dayofyear'] = test['pickup_datetime'].dt.dayofyear
def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))



train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, 

                                          train['dropoff_latitude'].values, train['dropoff_longitude'].values)



test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, 

                                         test['dropoff_latitude'].values, test['dropoff_longitude'].values)

def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

    a = haversine_array(lat1, lng1, lat1, lng2)

    b = haversine_array(lat1, lng1, lat2, lng1)

    return a + b



train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)





test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)







train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2

train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2

test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,

                    train[['dropoff_latitude', 'dropoff_longitude']].values,

                    test[['pickup_latitude', 'pickup_longitude']].values,

                    test[['dropoff_latitude', 'dropoff_longitude']].values))



pca = PCA().fit(coords)

train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]

train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]

train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]

test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]

test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]



train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])

test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])
sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])

train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])

test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])

t1 = dt.datetime.now()
fr1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', usecols=['id', 'total_distance', 'total_travel_time',  'number_of_steps', ])

fr2 = pd.read_csv('../input/new-york-city-taxi-with-osrm//fastest_routes_train_part_2.csv', usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

test_street_info = pd.read_csv('../input/new-york-city-taxi-with-osrm//fastest_routes_test.csv',

                               usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])
train_street_info = pd.concat((fr1, fr2))

train = train.merge(train_street_info, how='left', on='id')

test = test.merge(test_street_info, how='left', on='id')
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)
feature_names = list(train.columns)

print(np.setdiff1d(train.columns, test.columns))

do_not_use_for_training = ['id', 'log_trip_duration', 'trip_duration', 'dropoff_datetime', 'pickup_date', 

                           'pickup_datetime', 'date']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]

# print(feature_names)

print('We have %i features.' % len(feature_names))

train[feature_names].count()

         
train['store_and_fwd_flag'] = train['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)
test['store_and_fwd_flag'] = test['store_and_fwd_flag'].map(lambda x: 0 if x == 'N' else 1)
from sklearn.model_selection import KFold



X = train[feature_names].values

y = np.log(train['trip_duration'].values + 1)  





kf = KFold(n_splits=10)

kf.get_n_splits(X)



print(kf)  



KFold(n_splits=10, random_state=None, shuffle=False)

for train_index, test_index in kf.split(X):

    print("TRAIN:", train_index, "TEST:", test_index)

    X_train, X_test = X[train_index], X[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    

 
   

dtrain = xgb.DMatrix(X_train, label=y_train)

dvalid = xgb.DMatrix(X_test, label=y_test)

dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Try different parameters! My favorite is random search :)

xgb_pars = {'min_child_weight': 10, 'eta': 0.04, 'colsample_bytree': 0.8, 'max_depth': 15,

            'subsample': 0.75, 'lambda': 2, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1, 'gamma' : 0,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}    
model = xgb.train(xgb_pars, dtrain, 500, watchlist, early_stopping_rounds=250,

                  maximize=False, verbose_eval=15)