# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error as MSE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score, train_test_split

import numpy as np

import seaborn as sns



%matplotlib inline
test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')
train.head()
train.shape , test.shape
train.info()
train.isna().sum()
train.duplicated().sum()
train.isnull().sum()
plt.subplots(figsize=(18,7))

plt.title("Répartition des outliers")

train.boxplot()
train.loc[train.trip_duration<5000,"trip_duration"].hist(bins=120

                                                        )
#need it to be easier to loc 

train['log_trip_duration'] = np.log(train['trip_duration'].values)

train.log_trip_duration.hist(bins=100)
train = train[(train['log_trip_duration'] > 3.5)]

train = train[(train['log_trip_duration'] < 9)]
fig, ax = plt.subplots(figsize=(25, 20))

ax.scatter(train.pickup_longitude.values, train.pickup_latitude.values, s=5, color='black', alpha=0.5)

ax.set_xlim([-74.05, -74.00])

ax.set_ylim([40.70, 40.80])
train = train.loc[train['pickup_longitude']> -74.02]

train = train.loc[train['pickup_latitude']< 40.77]
fig, ax = plt.subplots(figsize=(25, 20))

ax.scatter(train.dropoff_longitude.values, train.dropoff_latitude.values, s=5, color='blue', alpha=0.5)

ax.set_xlim([-74.05, -74.00])

ax.set_ylim([40.70, 40.80])
train = train.loc[train['dropoff_longitude']> -74.02]

train = train.loc[train['dropoff_latitude']< 40.77]
train.passenger_count.value_counts().plot.bar()
train = train.loc[train['passenger_count']<6]

train = train[(train.passenger_count > 0)]
train.passenger_count.value_counts().plot.bar()
print("Min pickup time:",min(train['pickup_datetime']))

print("Max pickup time:",max(train['pickup_datetime']))
train['dist'] = np.sqrt(np.square(train['pickup_longitude'] - train['dropoff_longitude']) + np.square(train['pickup_latitude'] - train['dropoff_latitude']))

test['dist'] = np.sqrt(np.square(test['pickup_longitude'] - test['dropoff_longitude']) + np.square(test['pickup_latitude'] - test['dropoff_latitude']))



train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])



train['month'] = train['pickup_datetime'].dt.month

test['month'] = test['pickup_datetime'].dt.month



train['day'] = train['pickup_datetime'].dt.dayofweek

test['day'] = test['pickup_datetime'].dt.dayofweek



train['hour'] = train['pickup_datetime'].dt.hour

test['hour'] = test['pickup_datetime'].dt.hour
sns.barplot(x='hour',y='trip_duration',data=train)
train = train.loc[train['hour']>=8]
#SELECTION = ["passenger_count", "pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","dist","hour"]

#TARGET = "trip_duration"





#X_train = train[SELECTION]

#y_train = train[TARGET]

y = train["log_trip_duration"] # <-- target

X_train = train[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month","hour","day","dist"]] # <-- features



X_test = test[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","month","hour","day","dist"]]
xtrain, xvalid, ytrain, yvalid = train_test_split(X_train,y, test_size=0.2, random_state=42)

xtrain.shape, xvalid.shape, xtrain.shape, yvalid.shape
import lightgbm as lgb

dtrain = lgb.Dataset(X_train, y)



lgb_params = {

    'learning_rate': 0.1,

    'max_depth': 25,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.5,

    'num_leaves': 1000, 

    'objective': 'regression',

    'max_bin': 1000}
'''

sgd = SGDRegressor()

sgd.fit(X_train, y_train)

'''
'''

loss = MSE(y_train, sgd.predict(X_train))

loss

np.sqrt(np.log(loss))

'''
#SGD IS VERY BAD
'''

lr = LinearRegression()

lr.fit(X_train, y_train)

loss = MSE(y_train, lr.predict(X_train))

np.sqrt(np.log(loss))

'''
#LINEAR IS ALSO BAD
resultsCV = lgb.cv(lgb_params,dtrain,num_boost_round=100,nfold=3,metrics='mae',early_stopping_rounds=10,stratified=False)

print('best score :', resultsCV['l1-mean'][-1])
model_lgb = lgb.train(lgb_params, 

                      dtrain,

                      num_boost_round=1200)
#storing the predicitions



pred_test = np.exp(model_lgb.predict(X_test))

pred_test

submit = pd.read_csv('../input/sample_submission.csv')
submit.head()
submit['trip_duration'] = pred_test

submit.head(20)
submit_file = pd.DataFrame({"id": test.id, "trip_duration": pred_test})
submit_file.to_csv('submission.csv', index=False)