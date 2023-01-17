%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import timedelta

import datetime as dt

# from geopy.distance import vincenty, great_circle

from haversine import haversine

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')
print('We have {} training rows and {} test rows.'.format(train.shape[0], test.shape[0]))

print('We have {} training columns and {} test columns.'.format(train.shape[1], test.shape[1]))

train.head(2)
print('Id is unique.') if train.id.nunique() == train.shape[0] else print('oops')

print('Train and test sets are distinct.') if len(np.intersect1d(train.id.values, test.id.values))== 0 else print('oops')

print('We do not need to worry about missing values.') if train.count().min() == train.shape[0] and test.count().min() == test.shape[0] else print('oops')

print('The store_and_fwd_flag has only two values {}.'.format(str(set(train.store_and_fwd_flag.unique()) | set(test.store_and_fwd_flag.unique()))))
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)

test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)

train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)

train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')

test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')

train['check_trip_duration'] = (train['dropoff_datetime'] - train['pickup_datetime']).map(lambda x: x.total_seconds())

duration_difference = train[np.abs(train['check_trip_duration'].values  - train['trip_duration'].values) > 1]

print('Trip_duration and datetimes are ok.') if len(duration_difference[['pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration']]) == 0 else print('Ooops.')
def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    # calculate haversine

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
train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday

train.loc[:, 'pickup_day'] = train['pickup_datetime'].dt.day

train.loc[:, 'pickup_month'] = train['pickup_datetime'].dt.month

train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour

train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).map(lambda x: x.total_seconds())



test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date

test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday

test.loc[:, 'pickup_day'] = test['pickup_datetime'].dt.day

test.loc[:, 'pickup_month'] = test['pickup_datetime'].dt.month

test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).map(lambda x: x.total_seconds())
train.loc[:, 'average_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']

# train.loc[:, 'average_speed_v'] = 1000 * train['distance_vincenty'] / train[trip_duration]

# train.loc[:, 'average_speed_gc'] = 1000 * train['distance_great_circle'] / train[trip_duration]

train.loc[:, 'average_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
fig, ax = plt.subplots(ncols=3, sharey=True)

ax[0].plot(train.groupby('pickup_hour').mean()['average_speed_h'], 'bo-', alpha=0.5)

ax[1].plot(train.groupby('pickup_weekday').mean()['average_speed_h'], 'go-',  alpha=0.5)

ax[2].plot(train.groupby('pickup_day').mean()['average_speed_h'], 'ro',  alpha=0.5)

ax[0].set_xlabel('hour')

ax[1].set_xlabel('weekday')

ax[2].set_xlabel('day')

ax[0].set_ylabel('average speed')

fig.suptitle('Rush hour average traffic speed')

plt.show()
plt.plot(train.groupby('pickup_date').count()[['id']], 'o-', label='train')

plt.plot(test.groupby('pickup_date').count()[['id']], 'o-', label='test')

plt.title('Train and test period complete overlap.')

plt.legend(loc=0)

plt.ylabel('number of records')

plt.show()
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)

ax[0].plot(train['pickup_latitude'].values, train['pickup_longitude'].values, 'b.',

           label='train', alpha=0.1)

ax[1].plot(test['pickup_latitude'].values, test['pickup_longitude'].values, 'g.',

           label='train', alpha=0.1)

fig.suptitle('Train and test area complete overlap.')

ax[0].legend(loc=0)

ax[0].set_ylabel('latitude')

ax[0].set_xlabel('longitude')

ax[1].set_xlabel('longitude')

ax[1].legend(loc=0)

plt.xlim([40.5, 41])

plt.ylim([-74.5, -73.5])

plt.show()

for gby_col in ['pickup_hour', 'pickup_day', 'pickup_date', 'pickup_weekday']:

    gby = train.groupby(gby_col).mean()[['average_speed_h', 'average_speed_m']]

    gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]

    train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)

    test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)
train['trip_duration'].describe()

feature_names = list(train.columns)

do_not_use_for_training = ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration',

                           'check_trip_duration', 'pickup_date', 'average_speed_h', 'average_speed_m']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]

print(feature_names)

train[feature_names].count()



y = np.log(train['trip_duration'].values + 1)

plt.hist(y, bins=100)

plt.xlabel('log(trip_duration)')

plt.ylabel('number of train records')

plt.show()
Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 10, 'eta': 0.2, 'colsample_bytree': 0.5, 'max_depth': 10,

            'subsample': 0.95, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}

model = xgb.train(xgb_pars, dtrain, 400, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=50)
ypred = model.predict(dvalid)

plt.scatter(ypred, yv, alpha=0.1)

plt.xlabel('log(prediction)')

plt.ylabel('log(ground truth)')

plt.show()



plt.scatter(np.exp(ypred), np.exp(yv), alpha=0.1)

plt.xlabel('prediction')

plt.ylabel('ground truth')

plt.show()
ytest = model.predict(dtest)

print((test.shape, ytest.shape))

test['trip_duration'] = np.exp(ytest)

test[['id', 'trip_duration']].to_csv('first_submission.csv', index=False)