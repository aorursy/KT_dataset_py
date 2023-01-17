import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import multiprocessing

import warnings

import os

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
# read the data

#building is common on both train and test

# we have separate weather for both train and test

train_df=pd.read_csv("/kaggle/input/ashrae-energy-prediction/train.csv")

building_df=pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")

weather_train_df=pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

weather_test_df=pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")

test_df=pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")

print("Data readed successfully")
gc.collect()
# reduce the memory to join the

train_df=reduce_mem_usage(train_df,use_float16=True)

building_df=reduce_mem_usage(building_df,use_float16=True)

weather_train_df=reduce_mem_usage(weather_train_df,use_float16=True)

weather_test_df=reduce_mem_usage(weather_test_df,use_float16=True)

test_df=reduce_mem_usage(test_df,use_float16=True)
# we need to combine the different csv into single one for analysis

# for train  we need to combine the building metadata with building_id and site_id for the weather files.

# same as test

train_df=train_df.merge(building_df,on='building_id',how='left')

train_df=train_df.merge(weather_train_df,on=['site_id', 'timestamp'], how='left')

train_df.head()
# same for the test 

test_df=test_df.merge(building_df,on='building_id',how='left')

test_df=test_df.merge(weather_test_df,on=['site_id', 'timestamp'], how='left')

test_df.head()
train_df=reduce_mem_usage(train_df,use_float16=True)

test_df=reduce_mem_usage(test_df,use_float16=True)
# delete the variables for memory problem

gc.collect()

del building_df,weather_train_df,weather_test_df
# we have combine the data successfully how we can start to analyze the data

# we can start

print(train_df.columns)

print("The total number of columns",len(train_df.columns))
train_df.timestamp = pd.to_datetime(train_df.timestamp)

test_df.timestamp=pd.to_datetime(test_df.timestamp)
#its time series data we need to take care of 

train_df["hour"] = train_df["timestamp"].dt.hour

train_df["day"] = train_df["timestamp"].dt.day

train_df["weekend"] = train_df["timestamp"].dt.weekday

train_df["month"] = train_df["timestamp"].dt.month

train_df["dayofweek"] = train_df["timestamp"].dt.dayofweek

train_df["Year"] = train_df["timestamp"].dt.year
# test data

#its time series data we need to take care of 

test_df["hour"] = test_df["timestamp"].dt.hour

test_df["day"] = test_df["timestamp"].dt.day

test_df["weekend"] = test_df["timestamp"].dt.weekday

test_df["month"] = test_df["timestamp"].dt.month

test_df["dayofweek"] = test_df["timestamp"].dt.dayofweek

test_df["Year"] = test_df["timestamp"].dt.year
for cols in train_df.columns:

    if cols not in test_df.columns:

        print(cols)
# feature engineering

# need for the columns 

"""[ 'meter', 'site_id',

       'primary_use', 'square_feet', 'year_built', 'floor_count',

       'air_temperature', 'cloud_coverage', 'dew_temperature',

       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

       'wind_speed']"""

#air_temperature,dew_temperature,precip_depth_1_hr,sea_level_pressure,wind_direction,wind_speed

transform_col=['air_temperature','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed']



#train_df['air_temperature_mean']=train_df.groupby(['site_id','month','dayofweek'])['air_temperature'].transform('mean')

for col in transform_col:

    train_df[col+"_mean"]=train_df.groupby(['site_id','month','dayofweek'])[col].transform('mean')

    train_df[col+"_std"]=train_df.groupby(['site_id','month','dayofweek'])[col].transform('std')

    test_df[col+"_mean"]=test_df.groupby(['site_id','month','dayofweek'])[col].transform('mean')

    test_df[col+"_std"]=test_df.groupby(['site_id','month','dayofweek'])[col].transform('std')

    del train_df[col],test_df[col]

for col in train_df.columns:

    if col not in test_df.columns:

        print("Col:",col)





#train_df.groupby(['site_id','month','dayofweek'])['air_temperature'].transform('std')

train_df=reduce_mem_usage(train_df,use_float16=True)

test_df=reduce_mem_usage(test_df,use_float16=True)
gc.collect()

categorical_cols = ['site_id','meter','primary_use','hour','day','weekend','Year','month','dayofweek']

for c in categorical_cols:

    train_df[c]=pd.Categorical(train_df[c])

    test_df[c]=pd.Categorical(test_df[c])

# now we can bulid the model on the top of this dataframe
del train_df['building_id'],test_df['building_id']

del test_df['row_id']

for col in test_df.columns:

    if col not in train_df.columns:

        print("Missing:",col)
y=train_df['meter_reading']

del train_df['meter_reading']

gc.collect()

list1=train_df.columns
del train_df['timestamp'],test_df['timestamp']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(train_df,y,test_size=0.25,random_state=42)

print (x_train.shape)

print (y_train.shape)

print (x_test.shape)

print (y_test.shape)



lgb_train = lgb.Dataset(x_train, y_train,categorical_feature=categorical_cols)

lgb_test = lgb.Dataset(x_test, y_test,categorical_feature=categorical_cols)

del x_train, x_test , y_train, y_test



params = {'feature_fraction': 0.75,

          'bagging_fraction': 0.75,

          'objective': 'regression',

          'max_depth': -1,

          'learning_rate': 0.15,

        #  "boosting_type": "gbdt",

          "bagging_seed": 11,

          "metric": 'rmse',

          "verbosity": -1,

          'reg_alpha': 0.5,

          'reg_lambda': 0.5,

          'random_state': 47

         }



del train_df
reg = lgb.train(params, lgb_train, num_boost_round=3000, valid_sets=[lgb_test], early_stopping_rounds=100, verbose_eval = 100)

del lgb_train,lgb_test
len(reg.feature_importance())
#del list1[1]

li=list(list1)

del li[1]


ser = pd.DataFrame(reg.feature_importance(),li,columns=['Importance']).sort_values(by='Importance')

ser['Importance'].plot(kind='bar',figsize=(10,6))


predictions = []

#https://www.kaggle.com/kulkarnivishwanath/ashrae-great-energy-predictor-iii-eda-model

step = 50000

for i in range(0, len(test_df), step):

    print(str(i)+"Steps-----------------")

    predictions.extend(np.expm1(reg.predict(test_df.iloc[i: min(i+step, len(test_df)), :], num_iteration=reg.best_iteration)))
predictions[1:10]
sample_df=pd.read_csv("/kaggle/input/ashrae-energy-prediction/sample_submission.csv")
sample_df['meter_reading']=predictions

sample_df['meter_reading'].clip(lower=0,upper=None,inplace=True)
sample_df.head()
gc.collect()
sample_df.to_csv("lgbm_model_1.csv",index=None)