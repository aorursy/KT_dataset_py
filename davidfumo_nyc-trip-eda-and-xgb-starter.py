import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import xgboost as xgb

from sklearn.cross_validation import train_test_split

from sklearn import model_selection

from haversine import haversine

import datetime

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Load data

train = pd.read_csv('../input/train.csv', parse_dates=['pickup_datetime'])

test = pd.read_csv('../input/test.csv', parse_dates=['pickup_datetime']) #, parse_dates=['pickup_datetime']

sample_sub = pd.read_csv('../input/sample_submission.csv')
# how many instances and features do we have here?

print("Train shape: {}\nTest Shape: {}".format(train.shape, test.shape))
# train cols

print(list(train.columns))

print("--------------------------------")

print(list(test.columns))
train.head()
y_train = train['trip_duration']

train = train[test.columns].drop('id', axis=1)

test_ids = test['id']

test.drop('id', axis=1, inplace=True)



print("2nd Train shape: {}\n2nd Test Shape: {}".format(train.shape, test.shape))
# trip time 

#plt.figure(figsize=(8,6))

sns.distplot(y_train)

plt.ylabel('Trip duration in seconds')

plt.show()
# actualy the hist above is weird! maybe there are some uncommon values.

print("Min Trip time: {}\nMedian Trip Time: {}\nMax Trip Time: {}".format(y_train.min(), y_train.median(), y_train.max()))
# vendor_id distribuition

# vendor_id: a code indicating the provider associated with the trip record

plt.figure(figsize=(7,5))

sns.countplot(train['vendor_id'])
# passenger_count dist

# passenger_count(self-explanatory right?): the number of passengers in the vehicle

plt.figure(figsize=(7,5))

sns.countplot(train['passenger_count'])
# store_and_fwd_flag dist

# there is a big gap in the classes N and Y 

plt.figure(figsize=(7,5))

sns.countplot(train['store_and_fwd_flag'])
# pickup and dropoff

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)



# pickup_ Latitude vs Longitude

sns.regplot('pickup_longitude', 'pickup_latitude',

           data=train,

           fit_reg=False,

           scatter_kws={"marker": "D",

                        "s": 50}, ax=ax1)



# dropoff_longitude	vs dropoff_latitude

sns.regplot('dropoff_longitude', 'dropoff_latitude',

           data=train,

           fit_reg=False,

           scatter_kws={"marker": "D",

                        "s": 50}, ax=ax2)

# distance between pickup and dropoff

train['distance'] = train.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),

                                                    (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)

test['distance'] = test.apply(lambda x: haversine((x["pickup_longitude"], x["pickup_latitude"]),

                                                    (x["dropoff_longitude"], x["dropoff_latitude"])), axis=1)
# new train feats pickup (d h, m, s)

train['pickup_day'] = train['pickup_datetime'].apply(lambda x: x.day)

train['pickup_hour'] = train['pickup_datetime'].apply(lambda x: x.hour)

train['pickup_minute'] = train['pickup_datetime'].apply(lambda x: x.minute)

train['pickup_second'] = train['pickup_datetime'].apply(lambda x: x.second)



test['pickup_day'] = test['pickup_datetime'].apply(lambda x: x.day)

test['pickup_hour'] = test['pickup_datetime'].apply(lambda x: x.hour)

test['pickup_minute'] = test['pickup_datetime'].apply(lambda x: x.minute)

test['pickup_second'] = test['pickup_datetime'].apply(lambda x: x.second)



# convert categories to numbers

train['store_and_fwd_flag'] = train['store_and_fwd_flag'].replace({"N": 0, "Y": 1})

test['store_and_fwd_flag'] = test['store_and_fwd_flag'].replace({"N": 0, "Y": 1})
features_to_use = ['vendor_id', 'pickup_hour', 'pickup_minute', 'pickup_second', 'passenger_count',

                   'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'distance']
X_train = train[features_to_use].values

test_X = test[features_to_use].values
# evaluation function

# from : https://www.kaggle.com/lbronchal/xgboost-model-0-56?scriptVersionId=1360268

def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))



def xgb_rmsle_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'rmsle', rmsle(preds, labels)





# xgb training

def train_xgb(X, y, params, rounds, seed=0):

	print("Will train XGB for {} rounds".format(rounds))

	x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=seed)



	xg_train = xgb.DMatrix(x, label=y_train)

	xg_val = xgb.DMatrix(X_val, label=y_val)



	watchlist  = [(xg_train,'train'), (xg_val,'valid')]

	# return xgb.train(params, xg_train, rounds, watchlist, feval=xgb_rmsle_score) 

	return xgb.train(params, xg_train, rounds, watchlist, early_stopping_rounds=20) 



def predict_xgb(model, X_test):

	return model.predict(xgb.DMatrix(X_test))
xgb_params = {

    'eta': 0.03,

    'max_depth': 6,

    'subsample': 1,

    'colsample_bytree': 0.9,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'min_child_weight': 3,

    'silent': 1

}
model = train_xgb(X_train, y_train, xgb_params, 1000)
y_pred = predict_xgb(model, test_X)

out = pd.DataFrame({'id': test_ids, 'trip_duration': y_pred})

out.to_csv('xgb_starter.csv', index=False)

out.head()