# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# matplotlib and seaborn for plotting

import matplotlib.pyplot as plt



import seaborn as sns

import matplotlib.patches as patches



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

pd.set_option('max_columns', 150)



py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import os

import random

import math

import psutil

import pickle



from sklearn.model_selection import train_test_split,KFold

from sklearn.preprocessing import LabelEncoder
metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",

                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}

train_dtype = {'meter':"uint8",'building_id':'uint16'}
%%time



weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=['timestamp'], dtype=weather_dtype)

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)



metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)



train = pd.read_csv("../input/ashrae-energy-prediction/train.csv", parse_dates=['timestamp'], dtype=train_dtype)

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)



print('Size of train_df data', train.shape)

print('Size of weather_train_df data', weather_train.shape)

print('Size of weather_test_df data', weather_test.shape)

print('Size of building_meta_df data', metadata.shape)
train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
train.head()
weather_train.head()
# Dropping floor_count variable as it has 75% missing values

metadata.drop('floor_count',axis=1,inplace=True)
for df in [train, test]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")
train['meter_reading'] = np.log1p(train['meter_reading'])
metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",

                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",

                                "Utility":"Other","Religious worship":"Other"},inplace=True)

metadata['square_feet'] = np.log1p(metadata['square_feet'])

metadata.drop('year_built',axis=1,inplace=True) # delete instead of fill in gaps
%%time

train = pd.merge(train,metadata,on='building_id',how='left')

test  = pd.merge(test,metadata,on='building_id',how='left')

print ("Training Data+Metadata Shape {}".format(train.shape))

print ("Testing Data+Metadata Shape {}".format(test.shape))

gc.collect()

train = pd.merge(train,weather_train,on=['site_id','timestamp'],how='left')

test  = pd.merge(test,weather_test,on=['site_id','timestamp'],how='left')

print ("Training Data+Metadata+Weather Shape {}".format(train.shape))

print ("Testing Data+Metadata+Weather Shape {}".format(test.shape))

gc.collect()
# Save space

for df in [train,test]:

    df['square_feet'] = df['square_feet'].astype('float16')

    

# Fill NA

cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

for col in cols:

    train[col].fillna(np.nanmean(train[col].tolist()),inplace=True)

    test[col].fillna(np.nanmean(test[col].tolist()),inplace=True)

    

# Drop nonsense entries

# As per the discussion in the following thread, https://www.kaggle.com/c/ashrae-energy-prediction/discussion/117083, there is some discrepancy in the meter_readings for different ste_id's and buildings. It makes sense to delete them

idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp'] < "2016-05-21 00:00:00")]).index)

print (len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)



# dropping all the electricity meter readings that are 0, after considering them as anomalies.

idx_to_drop = list(train[(train['meter'] == "Electricity") & (train['meter_reading'] == 0)].index)

print(len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)
train.head()
# Not using constructed features, not dropping correlated ones
train.drop('timestamp',axis=1,inplace=True)

test.drop('timestamp',axis=1,inplace=True)



le = LabelEncoder()



train['meter']= le.fit_transform(train['meter']).astype("uint8")

test['meter']= le.fit_transform(test['meter']).astype("uint8")

train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")

test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")



print (train.shape, test.shape)
categorical_cols = ['building_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth']
from sklearn.ensemble import RandomForestRegressor as RF

import lightgbm as lgb



def train_site_model(train, site):

    print("Training model for site", site)

    

    train = train.loc[site]

    y = train['meter_reading']

    train.drop('meter_reading',axis=1,inplace=True)

    

    # 0 has too much missing data in certain months, so will throw away the month value to force it to consider temperature and the like for detecting winter times instead of using the month value

    if site == 0:

        train["Month"] = 0

    

    x_train,x_test,y_train,y_test = train_test_split(train,y,test_size=0.1,random_state=42)

    print (x_train.shape)

    print (y_train.shape)

    print (x_test.shape)

    print (y_test.shape)

    

    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_cols)

    lgb_test = lgb.Dataset(x_test, y_test, categorical_feature=categorical_cols)

    del x_train, x_test , y_train, y_test



    params = {'feature_fraction': 0.85, # 0.75

              'bagging_fraction': 0.75,

              'objective': 'regression',

               "num_leaves": 40, # New

              'max_depth': -1,

              'learning_rate': 0.15,

              "boosting_type": "gbdt",

              "bagging_seed": 11,

              "metric": 'rmse',

              "verbosity": -1,

              'reg_alpha': 0.5,

              'reg_lambda': 0.5,

              'random_state': 47

             }



    reg = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=100, verbose_eval=100)

    del lgb_train,lgb_test

    ser = pd.DataFrame(reg.feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')

    ser['Importance'].plot(kind='bar',figsize=(10,6))

    return reg
indexed_train = train.set_index(["site_id"])
models = []



for i in range(16):

    models.append(train_site_model(indexed_train, i))
Submission = pd.DataFrame(test.index, columns=['row_id'])

Submission['site_id'] = test['site_id']
test.set_index(['site_id'], inplace = True)

Submission.set_index(['site_id'], inplace=True)
%%time

# Couldn't just set them directly to the Submission dataframe because it complains about unmatching lengths, even though they are totally the same length grrr

Predictions = pd.Series(np.zeros(test.shape[0]), test.index)

for i in range(16):

    print("Predicting for site", i)

    Predictions.loc[i] = np.expm1(models[i].predict(test.loc[i]))

Submission['meter_reading'] = Predictions
Submission.head()
Submission['meter_reading'].clip(lower=0,upper=None,inplace=True)

Submission.to_csv("model_per_site.csv",index=None)
from IPython.display import FileLink

FileLink('model_per_site.csv')
Submission.shape