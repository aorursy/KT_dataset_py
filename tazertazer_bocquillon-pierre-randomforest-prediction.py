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

import numpy as np



%matplotlib inline
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_sample = pd.read_csv('../input/sample_submission.csv')
df.head(10)
df_test.head(10)
df.info()
df_test.info()
missing_col = list(set(df.columns).difference(set(df_test.columns)))

missing_col
is_dup = df.duplicated().sum()

is_null = df.isna().sum()

print("null : "+ str(is_null[::][1].sum()))

print("duplicated : "+ str(is_dup))
plt.subplots(figsize=(25,10))

df.boxplot()
fig, ax = plt.subplots(figsize=(25, 20))

ax.scatter(df.pickup_longitude.values, df.pickup_latitude.values, s=5, color='black', alpha=0.5)

ax.set_xlim([-74.05, -74.00])

ax.set_ylim([40.70, 40.80])
df = df.loc[df['pickup_longitude']> -74.02]

df = df.loc[df['pickup_latitude']< 40.77]
fig, ax = plt.subplots(figsize=(25, 20))

ax.scatter(df.pickup_longitude.values, df.pickup_latitude.values, s=5, color='black', alpha=0.5)

ax.set_xlim([-74.05, -74.00])

ax.set_ylim([40.70, 40.80])
fig, ax = plt.subplots(figsize=(25, 20))

ax.scatter(df.dropoff_longitude.values, df.dropoff_latitude.values, s=5, color='blue', alpha=0.5)

ax.set_xlim([-74.05, -74.00])

ax.set_ylim([40.70, 40.80])
df = df.loc[df['dropoff_longitude']> -74.02]

df = df.loc[df['dropoff_latitude']< 40.77]
fig, ax = plt.subplots(figsize=(25, 20))

ax.scatter(df.dropoff_longitude.values, df.dropoff_latitude.values, s=5, color='blue', alpha=0.5)

ax.set_xlim([-74.05, -74.00])

ax.set_ylim([40.70, 40.80])
plt.subplots(figsize=(25,10))

ax = df['passenger_count'].value_counts(normalize=True).plot.bar();

ax.set_ylabel("%")

ax.set_xlabel("passenger_count")
df = df[df['passenger_count'] > 0]

df = df[df['passenger_count'] <= 6]
fig, ax = plt.subplots(figsize=(25, 10))

df.loc[df.trip_duration<3600,"trip_duration"].hist(bins=50)
df['log_trip_duration'] = np.log(df['trip_duration'].values)
fig, ax = plt.subplots(figsize=(25, 10))

df.log_trip_duration.hist(bins=50)
df = df[(df['log_trip_duration'] > 3.5)]

df = df[(df['log_trip_duration'] < 9)]
fig, ax = plt.subplots(figsize=(25, 10))

df.log_trip_duration.hist(bins=50)
df['dist'] = np.sqrt(np.square(df['pickup_longitude'] - df['dropoff_longitude']) + np.square(df['pickup_latitude'] - df['dropoff_latitude']))

df_test['dist'] = np.sqrt(np.square(df_test['pickup_longitude'] - df_test['dropoff_longitude']) + np.square(df_test['pickup_latitude'] - df_test['dropoff_latitude']))
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df['m'] = df['pickup_datetime'].dt.month

df_test['m'] = df_test['pickup_datetime'].dt.month



df['d'] = df['pickup_datetime'].dt.dayofweek

df_test['d'] = df_test['pickup_datetime'].dt.dayofweek



df['h'] = df['pickup_datetime'].dt.hour

df_test['h'] = df_test['pickup_datetime'].dt.hour
ax = df['m'].value_counts().plot.bar()

ax.set_xlabel("month");

ax.set_ylabel("freq")
X_train = df[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","dist","m","h","d"]]

X_test = df_test[["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","dist","m","h","d"]]

y = df["log_trip_duration"]
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import lightgbm as lgb
xtrain, xvalid, ytrain, yvalid = train_test_split(X_train,y, test_size=0.2, random_state=1337)

xtrain.shape, xvalid.shape, xtrain.shape, yvalid.shape
dtrain = lgb.Dataset(X_train,y)
lgb_params = {

    'learning_rate': 0.1,

    'max_depth': 25,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.5,

    'num_leaves': 1000, 

    'objective': 'regression',

    'max_bin': 1000}
cv_results = lgb.cv(lgb_params,dtrain,num_boost_round=100,nfold=3,metrics='mae',early_stopping_rounds=10,stratified=False)
print('score :', cv_results['l1-mean'][-1])
df_test.head(10)
model_lgb = lgb.train(lgb_params,dtrain,num_boost_round=1200)
pred = np.exp(model_lgb.predict(X_test))

pred
df_sample['trip_duration'] = pred

submit = df_sample

submit.head(10)
submit_file = pd.DataFrame({"id": df_test.id, "trip_duration": pred})
submit_file.to_csv('submission.csv', index=False)