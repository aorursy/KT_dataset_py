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
import lightgbm as lgb

import matplotlib.pyplot as plt

import pandas as pd

#from sklearn.ensemble import RandomForestRegressor

#from sklearn.metrics import r2_score, mean_squared_error as MSE

#from sklearn.linear_model import SGDRegressor, LinearRegression

#from sklearn.model_selection import cross_val_score, train_test_split

import numpy as np





%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.head()
df_test.head()
df_train.info()
df_train.isna().sum()
df_train.duplicated().sum()
plt.subplots(figsize=(18,7))

plt.title("Gestion des outliers")

df_train.boxplot()
df_train.loc[df_train.trip_duration<5000,"trip_duration"].hist(bins=120)
df_train = df_train[(df_train['trip_duration'] > 60) & (df_train['trip_duration'] < 3600)]

df_train['trip_duration'] = np.log(df_train['trip_duration'].values)
from datetime import datetime
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])

df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime'])

df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
df_train['minute'] = df_train.pickup_datetime.dt.minute

df_train['hour'] = df_train.pickup_datetime.dt.hour

df_train['day'] = df_train.pickup_datetime.dt.dayofweek

df_train['month'] = df_train.pickup_datetime.dt.month



df_test['minute'] = df_test.pickup_datetime.dt.minute

df_test['hour'] = df_test.pickup_datetime.dt.hour

df_test['day'] = df_test.pickup_datetime.dt.dayofweek

df_test['month'] = df_test.pickup_datetime.dt.month
df_train['d_longitude'] = df_train['pickup_longitude'] - df_train['dropoff_longitude']

df_train['d_latitude'] = df_train['pickup_latitude'] - df_train['dropoff_latitude']



df_test['d_longitude'] = df_test['pickup_longitude'] - df_test['dropoff_longitude']

df_test['d_latitude'] = df_test['pickup_latitude'] - df_test['dropoff_latitude']



df_train['distance'] = np.sqrt(np.square(df_train['d_longitude']) + np.square(df_train['d_latitude']))

df_test['distance'] = np.sqrt(np.square(df_test['d_longitude']) + np.square(df_test['d_latitude']))
df_train.shape, df_test.shape
FEATURES = ["vendor_id","passenger_count","pickup_longitude", "pickup_latitude", "dropoff_longitude","dropoff_latitude","distance","month","hour","day"]

TARGET = "trip_duration"





X_train = df_train[FEATURES]

y_train = df_train[TARGET]
lgb_train = lgb.Dataset(X_train,y_train)
lgb_params = {

    'learning_rate': 0.1,

    'max_depth': 25,

    'num_leaves': 1000, 

    'objective': 'regression',

    'feature_fraction': 0.9,

    'bagging_fraction': 0.5,

    'max_bin': 1000}   
model_lgb = lgb.train(lgb_params,lgb_train,num_boost_round=500)
cv_score = lgb.cv(

        lgb_params,

        lgb_train,

        num_boost_round=100,

        nfold=3,

        metrics='mae',

        early_stopping_rounds=10,

        stratified=False

        )
print('Best CV score:', cv_score['l1-mean'][-1])
X_prediction = df_test[FEATURES]

prediction = np.exp(model_lgb.predict(X_prediction))

prediction
submission = pd.DataFrame({'id': df_test.id, 'trip_duration': prediction})

submission.to_csv('submission.csv', index=False)
submission.head()