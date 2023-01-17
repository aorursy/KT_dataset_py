%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from haversine import haversine
train = pd.read_csv("../input/train.csv")
train.head()
train.info()
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
test = pd.read_csv("../input/test.csv")
test.info()
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
train['distance'] = train.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),

                                                    (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)
test['distance'] = test.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),

                                                    (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)
def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))
train["pickup_weekday"] = train["pickup_datetime"].dt.weekday

test["pickup_weekday"] = test["pickup_datetime"].dt.weekday
train["pickup_hour"] = train["pickup_datetime"].dt.hour

test["pickup_hour"] = test["pickup_datetime"].dt.hour
train["pickup_month"] = train["pickup_datetime"].dt.month

test["pickup_month"] = test["pickup_datetime"].dt.month
train["store_and_fwd_flag"] = train["store_and_fwd_flag"].map(lambda x: int(x=='N'))

test["store_and_fwd_flag"] = test["store_and_fwd_flag"].map(lambda x: int(x=='N'))
train["weekend"] = train["pickup_weekday"].map(lambda x: int(x==5 or x==6))

test["weekend"] = test["pickup_weekday"].map(lambda x: int(x==5 or x==6))



feature_cols = ["vendor_id", "passenger_count", "pickup_weekday", "pickup_hour", "distance",

               "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",

               "store_and_fwd_flag"]



feature_cols = ["vendor_id", "passenger_count", "pickup_hour", "distance",

               "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",

               "store_and_fwd_flag", "weekend", "pickup_month"]



x_train = train[feature_cols]

y_train = train['trip_duration'].values

x_test = test[feature_cols]
import xgboost as xgb

from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=123456)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(x_test)



params = {}

params['objective'] = 'reg:linear'

params['eta'] = 0.02

params['max_depth'] = 7



def xgb_rmsle_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'rmsle', rmsle(preds, labels)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, 

                feval=xgb_rmsle_score, maximize=False, verbose_eval=10)
_ = xgb.plot_importance(clf)
p_test = clf.predict(d_test)



sub = pd.DataFrame()

sub['id'] = test['id']

sub['trip_duration'] = p_test

sub.to_csv('xgb.csv', index=False)