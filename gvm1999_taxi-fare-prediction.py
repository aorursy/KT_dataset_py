# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime

import xgboost

from xgboost import plot_importance

from matplotlib import pyplot

from sklearn.model_selection import cross_val_score,KFold, GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn-whitegrid')

from scipy.stats import skew

from collections import OrderedDict

import os

import gc

gc.collect()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):

    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)



def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    """

    Return distance along great radius between pickup and dropoff coordinates.

    """

    #Define earth radius (km)

    R_earth = 6371

    #Convert degrees to radians

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,

                                                             [pickup_lat, pickup_lon, 

                                                              dropoff_lat, dropoff_lon])

    #Compute distances along lat, lon dimensions

    dlat = dropoff_lat - pickup_lat

    dlon = dropoff_lon - pickup_lon

    

    #Compute haversine distance

    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2

    return 2 * R_earth * np.arcsin(np.sqrt(a))



def add_datetime_info(dataset):

    #Convert to datetime format

    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")

    

    dataset['hour'] = dataset.pickup_datetime.dt.hour

    dataset['day'] = dataset.pickup_datetime.dt.day

    dataset['month'] = dataset.pickup_datetime.dt.month

    dataset['weekday'] = dataset.pickup_datetime.dt.weekday

    dataset['year'] = dataset.pickup_datetime.dt.year

    

    return dataset



def sphere_dist_bear(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):

    """

    Return distance along great radius between pickup and dropoff coordinates.

    """

    #Define earth radius (km)

    R_earth = 6371

    #Convert degrees to radians

    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,

                                                             [pickup_lat, pickup_lon, 

                                                              dropoff_lat, dropoff_lon])

    #Compute distances along lat, lon dimensions

    dlat = dropoff_lat - pickup_lat

    dlon = pickup_lon - dropoff_lon

    

    #Compute bearing distance

    a = np.arctan2(np.sin(dlon * np.cos(dropoff_lat)),np.cos(pickup_lat) * np.sin(dropoff_lat) - np.sin(pickup_lat) * np.cos(dropoff_lat) * np.cos(dlon))

    return a
for train in pd.read_csv('/kaggle/input/train.csv',chunksize=20000000):

    break;



train = train[(train['pickup_longitude']!=0) & (train['dropoff_longitude']!=0) & (train['fare_amount']>0)]

test = pd.read_csv('/kaggle/input/test.csv')



train = add_datetime_info(train)

train['distance'] = sphere_dist(train['pickup_latitude'], train['pickup_longitude'], 

                                   train['dropoff_latitude'] , train['dropoff_longitude'])

test = add_datetime_info(test)

test['distance'] = sphere_dist(test['pickup_latitude'], test['pickup_longitude'], 

                                   test['dropoff_latitude'] , test['dropoff_longitude'])
train = train[train['fare_amount']>=2.50]

train = train[(train['distance']<1000) & (train['fare_amount']<50)]

train = train[((train['pickup_latitude']>39)&(train['pickup_latitude']<41))|((train['dropoff_latitude']>39)&(train['dropoff_latitude']<41))]

train = train[((train['pickup_longitude']<-73) & (train['dropoff_longitude']<-73))&((train['pickup_longitude']>-75.6) & (train['dropoff_longitude']>-75.6))]

train = train[(train['pickup_longitude'] != train['dropoff_longitude'])&(train['pickup_latitude'] != train['dropoff_latitude'])]

train = train[~train.index.isin(list(train[(train['distance']<1.5) & (train['fare_amount']>10)].index))]

train = train[~train.index.isin(list(train[(train['distance']<15) & (train['fare_amount']>40)].index))]

train = train[~train.index.isin(list(train[(train['distance']>50) & (train['fare_amount']<50)].index))]



df = train.drop(['key','pickup_datetime'], axis=1)

df_final = test.drop(['key','pickup_datetime'], axis=1)



y = df['fare_amount'].values

x = df.drop(['fare_amount'],axis=1)

x.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
import lightgbm as lgbm



params = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'nthread': 4,

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.8,

        'bagging_fraction' : 1,

        'max_bin' : 5000 ,

        'bagging_freq': 20,

        'colsample_bytree': 0.6,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1,

        'zero_as_missing': True,

        'seed':0,

        'num_rounds':50000

    }

train_set = lgbm.Dataset(X_train, y_train, silent=False,categorical_feature=['weekday','year','month'])

test_set = lgbm.Dataset(X_test, y_test, silent=False,categorical_feature=['weekday','year','month'])
model = lgbm.train(params, train_set = train_set, num_boost_round=10000,verbose_eval=500, valid_sets=test_set)
x_final = df_final.values

test_key = test['key']

prediction = model.predict(x_final,iteration=model.best_iteration)

submission = pd.DataFrame({

        "key": test_key,

        "fare_amount": prediction

})

submission.to_csv('taxi_fare_submission.csv',index=False)