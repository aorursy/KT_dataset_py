import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing, metrics

import xgboost as xgb

import lightgbm as lgb

from haversine import haversine

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)
train_df = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv', parse_dates=['pickup_datetime'])

test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv', parse_dates=['pickup_datetime'])

print("Train dataframe shape : ",train_df.shape)

print("Test dataframe shape : ", test_df.shape)
train_df.head()
test_df.head()
train_df['log_trip_duration'] = np.log1p(train_df['trip_duration'].values)



plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.log_trip_duration.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('log_trip_duration', fontsize=12)

plt.show()
(train_df['log_trip_duration'] > 12).sum()
null_count_df = train_df.isnull().sum(axis=0).reset_index()

null_count_df.columns = ['col_name', 'null_count']

null_count_df
null_count_df = test_df.isnull().sum(axis=0).reset_index()

null_count_df.columns = ['col_name', 'null_count']

null_count_df
train_df['pickup_date'] = train_df['pickup_datetime'].dt.date

test_df['pickup_date'] = test_df['pickup_datetime'].dt.date



cnt_srs = train_df['pickup_date'].value_counts()

plt.figure(figsize=(12,4))

ax = plt.subplot(111)

ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
cnt_srs = test_df['pickup_date'].value_counts()

plt.figure(figsize=(12,4))

ax = plt.subplot(111)

ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
# day of the month #

train_df['pickup_day'] = train_df['pickup_datetime'].dt.day

test_df['pickup_day'] = test_df['pickup_datetime'].dt.day



# month of the year #

train_df['pickup_month'] = train_df['pickup_datetime'].dt.month

test_df['pickup_month'] = test_df['pickup_datetime'].dt.month



# hour of the day #

train_df['pickup_hour'] = train_df['pickup_datetime'].dt.hour

test_df['pickup_hour'] = test_df['pickup_datetime'].dt.hour



# Week of year #

train_df["week_of_year"] = train_df["pickup_datetime"].dt.weekofyear

test_df["week_of_year"] = test_df["pickup_datetime"].dt.weekofyear



# Day of week #

train_df["day_of_week"] = train_df["pickup_datetime"].dt.weekday

test_df["day_of_week"] = test_df["pickup_datetime"].dt.weekday



# Convert to numeric #

map_dict = {'N':0, 'Y':1}

train_df['store_and_fwd_flag'] = train_df['store_and_fwd_flag'].map(map_dict)

test_df['store_and_fwd_flag'] = test_df['store_and_fwd_flag'].map(map_dict)
# drop off the variables which are not needed #

cols_to_drop = ['id', 'pickup_datetime', 'pickup_date']

train_id = train_df['id'].values

test_id = test_df['id'].values

train_y = train_df.log_trip_duration.values

train_X = train_df.drop(cols_to_drop + ['dropoff_datetime', 'trip_duration', 'log_trip_duration'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)
def runXGB(train_X, train_y, val_X, val_y, test_X, eta=0.05, max_depth=5, min_child_weight=1, subsample=0.8, colsample=0.7, num_rounds=8000, early_stopping_rounds=50, seed_val=2017):

    params = {}

    params["objective"] = "reg:linear"

    params['eval_metric'] = "rmse"

    params["eta"] = eta

    params["min_child_weight"] = min_child_weight

    params["subsample"] = subsample

    params["colsample_bytree"] = colsample

    params["silent"] = 1

    params["max_depth"] = max_depth

    params["seed"] = seed_val

    params["nthread"] = -1



    plst = list(params.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    xgval = xgb.DMatrix(val_X, label = val_y)

    xgtest = xgb.DMatrix(test_X)

    watchlist = [ (xgtrain,'train'), (xgval, 'test') ]

    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=20)



    pred_val = model.predict(xgval, ntree_limit=model.best_ntree_limit)

    pred_test = model.predict(xgtest, ntree_limit=model.best_ntree_limit)



    return pred_val, pred_test



def runLGB(train_X, train_y, val_X, val_y, test_X, eta=0.05, max_depth=5, min_child_weight=1, subsample=0.8, colsample=0.7, num_rounds=8000, early_stopping_rounds=50, seed_val=2017):

    params = {}

    params["objective"] = "regression"

    params['metric'] = "l2_root"

    params["learning_rate"] = eta

    params["min_child_weight"] = min_child_weight

    params["bagging_fraction"] = subsample

    params["bagging_seed"] = seed_val

    params["feature_fraction"] = colsample

    params["verbosity"] = 0

    params["max_depth"] = max_depth

    params["nthread"] = -1



    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label = val_y)

    model = lgb.train(params, lgtrain, num_rounds, valid_sets=lgval, early_stopping_rounds=early_stopping_rounds, verbose_eval=20)



    pred_val = model.predict(val_X, num_iteration=model.best_iteration)

    pred_test = model.predict(test_X, num_iteration=model.best_iteration)



    return pred_val, pred_test, model
# Increase the num_rounds parameter to a higher value (1000) and run the model #

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

cv_scores = []

pred_test_full = 0

pred_val_full = np.zeros(train_df.shape[0])

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_X.ix[dev_index], train_X.ix[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val, pred_test, model = runLGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=5, max_depth=8, eta=0.3)

    pred_val_full[val_index] = pred_val

    pred_test_full += pred_test

    cv_scores.append(np.sqrt(metrics.mean_squared_error(val_y, pred_val)))

print(cv_scores)

print("Mean CV score : ",np.mean(cv_scores))



pred_test_full = pred_test_full / 5.

pred_test_full = np.expm1(pred_test_full)

pred_val_full = np.expm1(pred_val_full)



# saving train predictions for ensemble #

train_pred_df = pd.DataFrame({'id':train_id})

train_pred_df['trip_duration'] = pred_val_full

train_pred_df.to_csv("train_preds_lgb_baseline.csv", index=False)



# saving test predictions for ensemble #

test_pred_df = pd.DataFrame({'id':test_id})

test_pred_df['trip_duration'] = pred_test_full

test_pred_df.to_csv("test_preds_lgb_baseline.csv", index=False)

# difference between pickup and dropoff latitudes #

train_df['lat_diff'] = train_df['pickup_latitude'] - train_df['dropoff_latitude']

test_df['lat_diff'] = test_df['pickup_latitude'] - test_df['dropoff_latitude']



# difference between pickup and dropoff longitudes #

train_df['lon_diff'] = train_df['pickup_longitude'] - train_df['dropoff_longitude']

test_df['lon_diff'] = test_df['pickup_longitude'] - test_df['dropoff_longitude']



## compute the haversine distance ##

#train_df['haversine_distance'] = train_df.apply(lambda row: haversine( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)

#test_df['haversine_distance'] = test_df.apply(lambda row: haversine( (row['pickup_latitude'], row['pickup_longitude']), (row['dropoff_latitude'], row['dropoff_longitude']) ), axis=1)



# get the pickup minute of the trip #

train_df['pickup_minute'] = train_df['pickup_datetime'].dt.minute

test_df['pickup_minute'] = test_df['pickup_datetime'].dt.minute



# get the absolute value of time #

train_df['pickup_dayofyear'] = train_df['pickup_datetime'].dt.dayofyear

test_df['pickup_dayofyear'] = test_df['pickup_datetime'].dt.dayofyear
# drop off the variables which are not needed #

cols_to_drop = ['id', 'pickup_datetime', 'pickup_date']

train_id = train_df['id'].values

test_id = test_df['id'].values

train_y = train_df.log_trip_duration.values

train_X = train_df.drop(cols_to_drop + ['dropoff_datetime', 'trip_duration', 'log_trip_duration'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)



# Increase the num_rounds parameter to a higher value (1000) and run the model #

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

cv_scores = []

pred_test_full = 0

pred_val_full = np.zeros(train_df.shape[0])

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_X.ix[dev_index], train_X.ix[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val, pred_test, model = runLGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=5, max_depth=8, eta=0.3)

    pred_val_full[val_index] = pred_val

    pred_test_full += pred_test

    cv_scores.append(np.sqrt(metrics.mean_squared_error(val_y, pred_val)))

print(cv_scores)

print("Mean CV score : ",np.mean(cv_scores))



pred_test_full = pred_test_full / 5.

pred_test_full = np.expm1(pred_test_full)

pred_val_full = np.expm1(pred_val_full)



# saving train predictions for ensemble #

train_pred_df = pd.DataFrame({'id':train_id})

train_pred_df['trip_duration'] = pred_val_full

train_pred_df.to_csv("train_preds_lgb.csv", index=False)



# saving test predictions for ensemble #

test_pred_df = pd.DataFrame({'id':test_id})

test_pred_df['trip_duration'] = pred_test_full

test_pred_df.to_csv("test_preds_lgb.csv", index=False)
def haversine_array(lat1, lng1, lat2, lng2):

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return h



def bearing_array(lat1, lng1, lat2, lng2):

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))



train_df['haversine_distance'] = haversine_array(train_df['pickup_latitude'].values, 

                                                     train_df['pickup_longitude'].values, 

                                                     train_df['dropoff_latitude'].values, 

                                                     train_df['dropoff_longitude'].values)

test_df['haversine_distance'] = haversine_array(test_df['pickup_latitude'].values, 

                                                    test_df['pickup_longitude'].values, 

                                                    test_df['dropoff_latitude'].values, 

                                                    test_df['dropoff_longitude'].values)



train_df['direction'] = bearing_array(train_df['pickup_latitude'].values, 

                                          train_df['pickup_longitude'].values, 

                                          train_df['dropoff_latitude'].values, 

                                          train_df['dropoff_longitude'].values)

test_df['direction'] = bearing_array(test_df['pickup_latitude'].values, 

                                         test_df['pickup_longitude'].values, 

                                         test_df['dropoff_latitude'].values, 

                                         test_df['dropoff_longitude'].values)
train_fr_part1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv', 

                             usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

train_fr_part2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv', 

                             usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

test_fr = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv', 

                             usecols=['id', 'total_distance', 'total_travel_time', 'number_of_steps'])

train_fr = pd.concat((train_fr_part1, train_fr_part2))



train_df = train_df.merge(train_fr, how='left', on='id')

test_df = test_df.merge(test_fr, how='left', on='id')



del train_fr_part1, train_fr_part2, train_fr, test_fr

import gc; gc.collect()
### some more new variables ###

train_df['pickup_latitude_round3'] = np.round(train_df['pickup_latitude'],3)

test_df['pickup_latitude_round3'] = np.round(test_df['pickup_latitude'],3)



train_df['pickup_longitude_round3'] = np.round(train_df['pickup_longitude'],3)

test_df['pickup_longitude_round3'] = np.round(test_df['pickup_longitude'],3)



train_df['dropoff_latitude_round3'] = np.round(train_df['dropoff_latitude'],3)

test_df['dropoff_latitude_round3'] = np.round(test_df['dropoff_latitude'],3)



train_df['dropoff_longitude_round3'] = np.round(train_df['dropoff_longitude'],3)

test_df['dropoff_longitude_round3'] = np.round(test_df['dropoff_longitude'],3)
# drop off the variables which are not needed #

cols_to_drop = ['id', 'pickup_datetime', 'pickup_date']

train_id = train_df['id'].values

test_id = test_df['id'].values

train_y = train_df.log_trip_duration.values

train_X = train_df.drop(cols_to_drop + ['dropoff_datetime', 'trip_duration', 'log_trip_duration'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)



# Increase the num_rounds parameter to a higher value (1000) and run the model #

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

cv_scores = []

pred_test_full = 0

pred_val_full = np.zeros(train_df.shape[0])

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_X.ix[dev_index], train_X.ix[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val, pred_test, model = runLGB(dev_X, dev_y, val_X, val_y, test_X, num_rounds=5, max_depth=8, eta=0.3)

    pred_val_full[val_index] = pred_val

    pred_test_full += pred_test

    cv_scores.append(np.sqrt(metrics.mean_squared_error(val_y, pred_val)))

print(cv_scores)

print("Mean CV score : ",np.mean(cv_scores))



pred_test_full = pred_test_full / 5.

pred_test_full = np.expm1(pred_test_full)

pred_val_full = np.expm1(pred_val_full)



# saving train predictions for ensemble #

train_pred_df = pd.DataFrame({'id':train_id})

train_pred_df['trip_duration'] = pred_val_full

train_pred_df.to_csv("train_preds_lgb.csv", index=False)



# saving test predictions for ensemble #

test_pred_df = pd.DataFrame({'id':test_id})

test_pred_df['trip_duration'] = pred_test_full

test_pred_df.to_csv("test_preds_lgb.csv", index=False)