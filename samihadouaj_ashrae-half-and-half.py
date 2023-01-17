import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns',500)

from sklearn import metrics

import time

import lightgbm as lgb

from sklearn.model_selection import train_test_split

import math

import gc
holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                "2019-01-01"]
## Memory optimization



# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16



from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=True):

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
cols = ['air_temperature','precip_depth_1_hr','cloud_coverage','dew_temperature','site_id','timestamp']
train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

weather_train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',usecols = cols)

building_meta_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
#train_df,weather_train_df,building_meta_df = reduce_mem_usage(train_df),reduce_mem_usage(weather_train_df),reduce_mem_usage(building_meta_df)

train = train_df.merge(building_meta_df, on='building_id', how='left')



train = train.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

del weather_train_df,building_meta_df

gc.collect();

train['timestamp'] = pd.to_datetime(train['timestamp'])
train = reduce_mem_usage(train)

train.tail()
train.shape
train.dtypes
le = LabelEncoder()

train['primary_use'] = le.fit_transform(train['primary_use'])
chtar = (train.shape[0]) //2
train['holidays'] = train['timestamp'].astype(str).isin(holidays)

train['holidays'] = train['holidays'].map({True:1,False:0})



train['holidays'] = train['holidays'].astype(np.int8)
train[(train['site_id'] ==0) & (train['meter'] ==0)& (train['building_id'] <104)][['timestamp','meter_reading']].set_index('timestamp').plot()
train[train['site_id'] ==1][['meter_reading']].max()
train['hour'] = train['timestamp'].dt.hour

train['month'] = train['timestamp'].dt.month

train['weekday'] = train['timestamp'].dt.weekday
train['hour'] = train['hour'].astype(np.int8)

train['month'] = train['month'].astype(np.int8)

train['weekday'] = train['weekday'].astype(np.int8)
train = train.drop(columns=['timestamp'])



X_half_1 =train[: chtar].drop(columns = ['meter_reading'])

X_half_2 = train[ chtar : ].drop(columns = ['meter_reading'])



y_half_1 = np.log1p(train[ chtar : ]['meter_reading'])

y_half_2 = np.log1p(train[ : chtar  ]['meter_reading'])

del train

gc.collect()
import lightgbm as lgb



categorical_features = ["building_id", "site_id", "meter", "primary_use"]





d_half_1 = lgb.Dataset(X_half_1, label=y_half_1, categorical_feature=categorical_features, free_raw_data=False)

d_half_2 = lgb.Dataset(X_half_2, label=y_half_2, categorical_feature=categorical_features, free_raw_data=False)





watchlist_1 = [d_half_1, d_half_2]

watchlist_2 = [d_half_2, d_half_1]



params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 40,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse"

}



print("Building model with first half and validating on second half:")

model_half_1 = lgb.train(params, train_set=d_half_1, num_boost_round=1000, valid_sets=watchlist_1, verbose_eval=200, early_stopping_rounds=200)



print("Building model with second half and validating on first half:")

model_half_2 = lgb.train(params, train_set=d_half_2, num_boost_round=1000, valid_sets=watchlist_2, verbose_eval=200, early_stopping_rounds=200)
del X_half_1,X_half_2,y_half_1,y_half_2
gc.collect()
test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv', usecols = cols)
building_meta_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
test = test_df.merge(building_meta_df, on='building_id', how='left')

test = test.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

test["timestamp"] = pd.to_datetime(test["timestamp"])

del watchlist_1,watchlist_2
del test_df,train_df,weather_test_df,building_meta_df

gc.collect()
test['hour'] = test['timestamp'].dt.hour

test['month'] = test['timestamp'].dt.month

test['weekday'] = test['timestamp'].dt.weekday



test['hour'] = test['hour'].astype(np.int8)

test['month'] = test['month'].astype(np.int8)

test['weekday'] = test['weekday'].astype(np.int8)


test['holidays'] = test['timestamp'].astype(str).isin(holidays)

test['holidays'] = test['holidays'].map({True:1,False:0})

test['holidays'] = test['holidays'].astype(np.int8)

test = test.drop(columns=['timestamp'])
test = reduce_mem_usage(test)
rows_id = test['row_id']

test.drop(columns=['row_id'],inplace = True)
le = LabelEncoder()

test['primary_use'] = le.fit_transform(test['primary_use'])
preds_half1 = model_half_1.predict(test)
del model_half_1

gc.collect()
gc.collect()
preds_half2 = model_half_2.predict(test)
preds_final = np.expm1(preds_half1) + np.expm1(preds_half2) 
preds_final
sub = pd.DataFrame({"row_id":rows_id,"meter_reading":preds_final})

sub.to_csv('submission.csv', index=False)
from IPython.display import FileLink
FileLink('submission.csv')