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



import os

import random

import math

import psutil

import pickle



#conda install -c conda-forge lightgbm

from sklearn.ensemble import RandomForestRegressor as RF

import lightgbm as lgb



from sklearn.model_selection import train_test_split,KFold

from sklearn.preprocessing import LabelEncoder



metadata_dtype = {'site_id':"uint8",'building_id':'uint16','square_feet':'float32','year_built':'float32','floor_count':"float16"}

weather_dtype = {"site_id":"uint8",'air_temperature':"float16",'cloud_coverage':"float16",'dew_temperature':"float16",'precip_depth_1_hr':"float16",

                 'sea_level_pressure':"float32",'wind_direction':"float16",'wind_speed':"float16"}

train_dtype = {'meter':"uint8",'building_id':'uint16'}



weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv", parse_dates=['timestamp'], dtype=weather_dtype)

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv", parse_dates=['timestamp'], dtype=weather_dtype)



metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv", dtype=metadata_dtype)



train = pd.read_csv("../input/ashrae-energy-prediction/train.csv", parse_dates=['timestamp'], dtype=train_dtype)

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv", parse_dates=['timestamp'], usecols=['building_id','meter','timestamp'], dtype=train_dtype)



train['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)

test['meter'].replace({0:"Electricity",1:"ChilledWater",2:"Steam",3:"HotWater"},inplace=True)
#Drop column floor_count, more than 3/4 are missing observations

metadata.drop('floor_count',axis=1,inplace=True)
#Construct Month, Week, Hour features:

for df in [train, test]:

    df['Month'] = df['timestamp'].dt.month.astype("uint8")

    df['DayOfMonth'] = df['timestamp'].dt.day.astype("uint8")

    df['DayOfWeek'] = df['timestamp'].dt.dayofweek.astype("uint8")

    df['Hour'] = df['timestamp'].dt.hour.astype("uint8")

#Transform target variable. The idea is to squeeze outliers out

train['meter_reading'] = np.log1p(train['meter_reading'])
#As per visualisation made in https://www.kaggle.com/nroman/eda-for-ashrae, one might suspect seasonality inherent in data. Create season variable:

for df in [train, test]:

    df['Season'] = ((df['Month']%12+3)//3).astype("uint8")


metadata['primary_use'].replace({"Healthcare":"Other","Parking":"Other","Warehouse/storage":"Other","Manufacturing/industrial":"Other",

                                "Retail":"Other","Services":"Other","Technology/science":"Other","Food sales and service":"Other",

                                "Utility":"Other","Religious worship":"Other"},inplace=True)

metadata['square_feet'] = np.log(1+0.0003*metadata['square_feet'])

metadata['year_built'].fillna(-999, inplace=True)

metadata['year_built'] = metadata['year_built'].astype('int16')





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

#Let's save some space:

for df in [train,test]:

    df['square_feet'] = df['square_feet'].astype('float16')
#Encode meter and primary_use features. 

le = LabelEncoder()

train['primary_use']= le.fit_transform(train['primary_use']).astype("uint8")

test['primary_use']= le.fit_transform(test['primary_use']).astype("uint8")

train['meter']= le.fit_transform(train['meter']).astype("uint8")

test['meter']= le.fit_transform(test['meter']).astype("uint8")

#We will add interaction terms between season and some of the features:

#Add interaction between primary_use and seasonality

for df in [train,test]:

    df['primary_use_X_season']=df['primary_use']*df['Season']

    df['primary_use_X_season']=df['primary_use_X_season'].astype("uint8")



#Add interaction between building_id and seasonality

for df in [train,test]:

    df['building_id_X_season']=df['building_id']*df['Season']

    df['building_id_X_season']=df['building_id_X_season'].astype("uint8")



#Add interaction between meter and seasonality

for df in [train,test]:

    df['meter_X_season']=df['meter']*df['Season']

    df['meter_X_season']=df['meter_X_season'].astype("uint8")



#Add interaction between site_id and seasonality

for df in [train,test]:

    df['site_id_X_season']=df['site_id']*df['Season']

    df['site_id_X_season']=df['site_id_X_season'].astype("uint8")
#Replace Missing values:

cols = ['air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']

for col in cols:

    train[col].fillna(np.nanmean(train[col].tolist()),inplace=True)

    test[col].fillna(np.nanmean(test[col].tolist()),inplace=True)

#Let's drop entries which seem nonsense as per discussion given in https://www.kaggle.com/robertobianco/time-series-study

idx_to_drop = list((train[(train['site_id'] == 0) & (train['timestamp'] < "2016-05-21 00:00:00")]).index)

print (len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)



# dropping all the electricity meter readings that are 0, after considering them as anomalies.

idx_to_drop = list(train[(train['meter'] == "Electricity") & (train['meter_reading'] == 0)].index)

print(len(idx_to_drop))

train.drop(idx_to_drop,axis='rows',inplace=True)

#Drop timestamp feature:

train.drop('timestamp',axis=1,inplace=True)

test.drop('timestamp',axis=1,inplace=True)

#Split the data by train and validation splits:

y = train['meter_reading']

train.drop('meter_reading',axis=1,inplace=True)



X_1_1 = train[:int(train.shape[0] / 2)]

X_1_2 = train[int(train.shape[0] / 2):]







y_1_1 = y[:int(train.shape[0] / 2)]

y_1_2 = y[int(train.shape[0] / 2):]





categorical_cols = ['building_id_X_season','site_id_X_season','meter_X_season','primary_use_X_season','Season','building_id','site_id','Month','meter','Hour','primary_use','DayOfWeek','DayOfMonth']





lgb_1= lgb.Dataset(X_1_1, y_1_1, categorical_feature=categorical_cols,free_raw_data=False)

lgb_2 = lgb.Dataset(X_1_2, y_1_2, categorical_feature=categorical_cols,free_raw_data=False)



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

validation_1 = [lgb_1, lgb_2]

validation_2 = [lgb_2, lgb_1]

reg_1 = lgb.train(params, lgb_1, num_boost_round=1000, valid_sets=validation_1, verbose_eval=200, early_stopping_rounds=200)

reg_2 = lgb.train(params, lgb_2, num_boost_round=1000, valid_sets=validation_2, verbose_eval=200, early_stopping_rounds=200)

#Feature importance in each model:

ser = pd.DataFrame(reg_1.feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')

ser['Importance'].plot(kind='bar',figsize=(10,6))

plt.title("Feature Importance for first model")



ser = pd.DataFrame(reg_2.feature_importance(),train.columns,columns=['Importance']).sort_values(by='Importance')

ser['Importance'].plot(kind='bar',figsize=(10,6))

plt.title("Feature Importance for second model")



del X_1_1,X_1_2,y_1_1,y_1_2,weather_train,weather_test, metadata

del lgb_1,lgb_2

del train

gc.collect()
predictions = (reg_1.predict(test, num_iteration=reg_1.best_iteration)) / 2



del reg_1

gc.collect()





predictions += (reg_2.predict(test, num_iteration=reg_2.best_iteration)) / 2

    



del reg_2

gc.collect()





predictions=np.expm1(predictions)
Submission = pd.DataFrame(test.index,columns=['row_id'])

Submission['meter_reading'] = predictions

Submission['meter_reading'].clip(lower=0,upper=None,inplace=True)

Submission.to_csv("submission_3.csv",index=None)


