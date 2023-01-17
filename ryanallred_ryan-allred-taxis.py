import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import xgboost as xgb

import seaborn as sns

from sklearn.model_selection import train_test_split

import datetime as dt
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files 

# in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

t0 = pd.datetime.now()

np.random.seed(42)

train.head(2)
# Look at column types

print(train.dtypes)
# Check Shape

print('train shape: ', train.shape)

print('test shape: ', test.shape)



# Looks like the test dataset doesn't contain dropoff_datetime or trip_duration
# Check for missing values

print(train.isnull().sum())
# What are the values for store_and_fwd_flag ?

train.store_and_fwd_flag.unique()
# Make Binary store_and_fwd_flag

train["store_and_fwd_flag"].replace(('Y', 'N'), (1, 0), inplace=True)

test["store_and_fwd_flag"].replace(('Y', 'N'), (1, 0), inplace=True)



train.head(2)
plt.hist(train['trip_duration'].values, bins=100)

plt.xlabel('trip_duration')

plt.ylabel('number of train records')

plt.show()

print('Hmmm... That\'s not a very helpful graph... outliers? ')
# What's the max trip duration? we've got some outliers.

print("Trip Duration Min (seconds):", train['trip_duration'].min())

print("Trip Duration Max (seconds):", train['trip_duration'].max())

print("Max Trip Duration in Hours:", train['trip_duration'].max()/3600)
# Plot log of trip duration

train['log_trip_duration'] = np.log(train['trip_duration'])

plt.hist(train['log_trip_duration'].values, bins=100)

plt.xlabel('log_trip_duration')

plt.ylabel('number of train records')

#plt.show()
# # Haversine Formula for distance between two GPS Coordinates



# # from math import radians, cos, sin, asin, sqrt



# # def haversine(lon1, lat1, lon2, lat2):    

# #     """

# #     Calculate the great circle distance between two points 

# #     on the earth (specified in decimal degrees)

# #     Can verify values here: http://www.movable-type.co.uk/scripts/latlong.html

# #     """

# #     # convert decimal degrees to radians 

# #     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])



# #     # haversine formula 

# #     dlon = lon2 - lon1 

# #     dlat = lat2 - lat1 

# #     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

# #     c = 2 * asin(sqrt(a)) 

# #     r = 6371 # Radius of earth in kilometers. Use 3956 for miles

# #     return c * r



# # train['haversine_dist_km'] = train.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)

# # test['haversine_dist_km'] = train.apply(lambda row: haversine(row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']), axis=1)



# # train.head(2)



# def haversine_array(lat1, lng1, lat2, lng2):

#     lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

#     AVG_EARTH_RADIUS = 6371  # in km

#     lat = lat2 - lat1

#     lng = lng2 - lng1

#     d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

#     h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

#     return h



# def dummy_manhattan_distance(lat1, lng1, lat2, lng2):

#     a = haversine_array(lat1, lng1, lat1, lng2)

#     b = haversine_array(lat1, lng1, lat2, lng1)

#     return a + b



# def bearing_array(lat1, lng1, lat2, lng2):

#     AVG_EARTH_RADIUS = 6371  # in km

#     lng_delta_rad = np.radians(lng2 - lng1)

#     lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

#     y = np.sin(lng_delta_rad) * np.cos(lat2)

#     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

#     return np.degrees(np.arctan2(y, x))



# train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

# train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

# train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

# # train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])



# test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

# test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

# test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

# # test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])



# train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2

# train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

# test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2

# test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2
# # Speed = distance / time

# train.loc[:, 'avg_speed_h'] = 1000 * train['haversine_dist_km'] / train['trip_duration']
# # Datetime Features



# train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

# test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

# train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

# test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

# train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

# train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')

# test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

# train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime']).map(lambda x: x.total_seconds())

# duration_difference = train[np.abs(train['check_trip_duration'].values  - train['trip_duration'].values) > 1]

# print('Trip_duration and datetimes are ok.') if len(duration_difference[['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration']]) == 0 else print('Ooops.')



# train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday

# train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear

# train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour

# train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

# train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

# train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']



# test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday

# test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear

# test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

# test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

# test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

# test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']
feature_names = list(train.columns)

print("difference in datasets", np.setdiff1d(train.columns, test.columns))

#do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',

#                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',

#                           'center_lat_bin', 'center_long_bin', 'pickup_dt_bin', 'pickup_datetime_group', 'dropoff_datetime',

#                          'dropoff_day', 'dropoff_hour', 'dropoff_month', 'dropoff_second', 'dropoff_year']

do_not_use_for_training = ['id', 'dropoff_datetime', 'log_trip_duration', 'trip_duration', 'pickup_datetime']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]

print(feature_names)

print('We have %i features.' % len(feature_names))

train[feature_names].count()

y = np.log(train['trip_duration'].values + 1)



t1 = dt.datetime.now()

print('Feature extraction time: %i seconds' % (t1 - t0).seconds)



# train.head()
Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Try different parameters! My favorite is random search :)

xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}
# You could try to train with more epoch

model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=10)
print('Modeling RMSLE %.5f' % model.best_score)

t1 = dt.datetime.now()

print('Training time: %i seconds' % (t1 - t0).seconds)
ypred = model.predict(dvalid)

ytest = model.predict(dtest)

print('Test shape OK.') if test.shape[0] == ytest.shape[0] else print('Oops')

test['trip_duration'] = np.exp(ytest) - 1

test[['id', 'trip_duration']].to_csv('taxi_trip_3.csv.gz', index=False, compression='gzip')



print('Valid prediction mean: %.3f' % ypred.mean())

print('Test prediction mean: %.3f' % ytest.mean())



fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

sns.distplot(ypred, ax=ax[0], color='blue', label='validation prediction')

sns.distplot(ytest, ax=ax[1], color='green', label='test prediction')

ax[0].legend(loc=0)

ax[1].legend(loc=0)

plt.show()



t1 = dt.datetime.now()

print('Total time: %i seconds' % (t1 - t0).seconds)