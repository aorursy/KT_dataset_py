import pandas as pd

import numpy as np

import mxnet as mx

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from __future__ import print_function

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

import xgboost as xgb

import scipy as sc



N = 100000 # number of sample rows in plots

np.random.seed(1337)
test = pd.read_csv('test.csv', parse_dates=['pickup_datetime'])

train = pd.read_csv('train.csv', parse_dates=['pickup_datetime', 'dropoff_datetime'])



print('Test shape : {}'.format(test.shape))

print('Train shape : {}'.format(train.shape))

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



def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))
## Fix up store and fwd flag

train['store_and_fwd_flag'] = 1 * (train.store_and_fwd_flag.values == 'Y')

test['store_and_fwd_flag'] = 1 * (test.store_and_fwd_flag.values == 'Y')



## PCA 

full = pd.concat([train, test])

coords = np.vstack((full[['pickup_latitude', 'pickup_longitude']], 

                   full[['dropoff_latitude', 'dropoff_longitude']]))



pca = PCA().fit(coords)

train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]

train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]

train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]

test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]

test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
## Distance feature calculations



train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)

train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])



test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])



train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2

train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2

test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2











## Temporal features



train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday

train.loc[:, 'pickup_hour_weekofyear'] = train['pickup_datetime'].dt.weekofyear

train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour

train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute

train.loc[:, 'pickup_dt'] = (train['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

train.loc[:, 'pickup_week_hour'] = train['pickup_weekday'] * 24 + train['pickup_hour']



train.loc[:, 'pickup_date'] = train['pickup_datetime'].dt.date

test.loc[:, 'pickup_date'] = test['pickup_datetime'].dt.date



test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday

test.loc[:, 'pickup_hour_weekofyear'] = test['pickup_datetime'].dt.weekofyear

test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour

test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute

test.loc[:, 'pickup_dt'] = (test['pickup_datetime'] - train['pickup_datetime'].min()).dt.total_seconds()

test.loc[:, 'pickup_week_hour'] = test['pickup_weekday'] * 24 + test['pickup_hour']
## Speed Features



train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']

# train.loc[:, 'average_speed_v'] = 1000 * train['distance_vincenty'] / train[trip_duration]

# train.loc[:, 'average_speed_gc'] = 1000 * train['distance_great_circle'] / train[trip_duration]

train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
## Binning



train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 3)

train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 3)

# Average speed for regions

gby_cols = ['pickup_lat_bin', 'pickup_long_bin']

coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()

coord_count = train.groupby(gby_cols).count()[['id']].reset_index()

coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)

coord_stats = coord_stats[coord_stats['id'] > 100]

fig, ax = plt.subplots(ncols=1, nrows=1)

ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], color='black', s=0.02, alpha=0.05)

ax.scatter(coord_stats.pickup_long_bin.values, coord_stats.pickup_lat_bin.values, c=coord_stats.avg_speed_h.values,

           cmap='RdYlGn', s=10, alpha=0.3, vmin=0, vmax=8)

ax.set_xlim(-74.03, -73.77)

ax.set_ylim(40.63, 40.85)

ax.set_xlabel('Longitude')

ax.set_ylabel('Latitude')

fig.savefig('coords.png', figsize=(16, 10), dpi=300)

plt.title('Average speed')

plt.show()



train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)

train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)

train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)

train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)

train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))

test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)

test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)

test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)

test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)

test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)



for gby_col in ['pickup_hour', 'pickup_date', 'pickup_weekday', 'pickup_dt_bin',

               'pickup_week_hour']:

    gby = train.groupby(gby_col).mean()[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']]

    gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]

    train = pd.merge(train, gby, how='left', left_on=gby_col, right_index=True)

    test = pd.merge(test, gby, how='left', left_on=gby_col, right_index=True)



for gby_cols in [['center_lat_bin', 'center_long_bin'],

                 ['pickup_hour', 'center_lat_bin', 'center_long_bin']] :

    coord_speed = train.groupby(gby_cols).mean()[['avg_speed_h']].reset_index()

    coord_count = train.groupby(gby_cols).count()[['id']].reset_index()

    coord_stats = pd.merge(coord_speed, coord_count, on=gby_cols)

    coord_stats = coord_stats[coord_stats['id'] > 100]

    coord_stats.columns = gby_cols + ['avg_speed_h_%s' % '_'.join(gby_cols), 'cnt_%s' %  '_'.join(gby_cols)]

    train = pd.merge(train, coord_stats, how='left', on=gby_cols)

    test = pd.merge(test, coord_stats, how='left', on=gby_cols)
feature_names = list(train.columns)

do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration',

                           'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin',

                           'center_lat_bin', 'center_long_bin', 'pickup_dt_bin']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]

print(feature_names)

print('We have %i features.' % len(feature_names))

train[feature_names].count()
y = np.log(train['trip_duration'].values + 1)



train[feature_names] = train[feature_names].fillna(0)

test[feature_names] = test[feature_names].fillna(0)

scaler = StandardScaler()

scaled_X = scaler.fit_transform(train[feature_names].values)

Xtr, Xv, ytr, yv = train_test_split(scaled_X, y, test_size=0.2, random_state=1982)
batch_size=2000



train_iter = mx.io.NDArrayIter(Xtr, ytr, batch_size, shuffle=True, label_name='lin_reg_label')

val_iter = mx.io.NDArrayIter(Xv, yv, batch_size, shuffle=False)
print("Building model...")

net = mx.sym.Variable('data')

Y = mx.symbol.Variable('lin_reg_label')

net = mx.sym.FullyConnected(net, name='fc1', num_hidden=1024)

net = mx.sym.Activation(net, name='relu1', act_type="relu")

net = mx.sym.Dropout(net, p=0.1)

net = mx.sym.FullyConnected(net, name='fc2', num_hidden=256)

net = mx.sym.Activation(net, name='relu2', act_type="relu")

net = mx.sym.Dropout(net, p=0.1)

net = mx.sym.FullyConnected(net, name='fc3', num_hidden=1024)

net = mx.sym.Activation(net, name='relu3', act_type="relu")

net = mx.sym.FullyConnected(net, name='fc4', num_hidden=1)

net = mx.sym.LinearRegressionOutput(net, label=Y, name='lro')



mx.viz.plot_network(net)
print("Training model")

num_epoch = 400

learning_rate = 0.25

momentum = 0.002



import logging

logging.getLogger().setLevel(logging.DEBUG)



mod = mx.mod.Module(symbol=net, label_names=['lin_reg_label'], context=mx.cpu())

mod.fit(

    train_iter,

    eval_data=val_iter,

    optimizer='nag',

    optimizer_params={'learning_rate':learning_rate, 'momentum':momentum},

    eval_metric='rmse',

    batch_end_callback = mx.callback.Speedometer(batch_size, 500), 

    num_epoch=num_epoch

)



mod.score(val_iter, ['rmse', 'acc'])
## Add in NN-predicted feature and retrain using XGB

scaled_nn_train = scaler.fit_transform(train[feature_names].values)

scaled_nn_test = scaler.fit_transform(test[feature_names].values)



nn_train_iter = mx.io.NDArrayIter(scaled_nn_train, None, batch_size)

nn_test_iter = mx.io.NDArrayIter(scaled_nn_test, None, batch_size)



train['nn_preds'] = np.exp(mod.predict(nn_train_iter).asnumpy())

test['nn_preds'] = np.exp(mod.predict(nn_test_iter).asnumpy())



feature_names.append('nn_preds')

Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Try different parameters

xgb_pars = {'min_child_weight': 150, 'learning_rate': 0.1, 'eta': 0.3, 'colsample_bytree': 0.7, 'max_depth': 14,

            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}
# You could try to train with more epoch

model = xgb.train(xgb_pars, dtrain, 100, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=20)

print('Modeling RMSLE %.5f' % model.best_score)

xgb_preds = np.exp(model.predict(dtest))



# Create file from predictions.

test['trip_duration'] = (test['nn_preds'] + xgb_preds) / 2

test[['id', 'trip_duration']].to_csv('submission.csv', index=False)