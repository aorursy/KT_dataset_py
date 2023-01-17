# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



# for plotting 

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")

weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")

train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
print('Size of train data', train.shape)

print('Size of weather_train data', weather_train.shape)

print('Size of building_meta data', building_metadata.shape)
## Function to reduce the DF size

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
## REducing memory

for df in [train, weather_train, building_metadata]:

    df = reduce_mem_usage(df)
train.head(3)

# print(train.dtypes)
train['meter_reading'].hist(figsize=(6, 5))
np.log1p(train['meter_reading']).hist(figsize=(6, 5))
weather_train.head(3)

# print(weather_train.dtypes)
weather_train["air_temperature"].hist(figsize=(6, 4))
weather_train["cloud_coverage"].hist(figsize=(6, 4))
weather_train["dew_temperature"].hist(figsize=(6, 4))
weather_train["sea_level_pressure"].hist(figsize=(6, 4))
weather_train["wind_speed"].hist(figsize=(6, 4))
building_metadata.head(3)

print(building_metadata.dtypes)
building_metadata["square_feet"].hist(figsize=(6, 4))
building_metadata["square_feet"] = building_metadata["square_feet"].apply(np.log1p)

building_metadata["square_feet"].hist(figsize=(6, 4))
building_metadata["primary_use"].value_counts()
def check_missing(df, ascending=False):

    total = df.isnull().sum().sort_values(ascending = ascending)

    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = ascending)

    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    

    # Only want to check columns with null values

    missing_data = missing_data[missing_data['Total']!=0]

    return missing_data
check_missing(train).head(len(train))
check_missing(weather_train).head(len(weather_train))
weather_train.drop(['cloud_coverage', 'precip_depth_1_hr'], axis=1, inplace=True)

weather_train.replace('NaN', np.nan, inplace=True)

for col in ["air_temperature", "dew_temperature", "sea_level_pressure", "wind_direction", "wind_speed"]:

    weather_train[col].fillna(weather_train[col].mean(), inplace=True)
check_missing(building_metadata).head(len(building_metadata))
building_metadata.drop(['floor_count', 'year_built'], axis=1, inplace=True)
def merging(df, weather):

    df = df.merge(building_metadata, left_on = "building_id", right_on = "building_id", how = "left")

    df = df.merge(weather, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")

    return df
train = merging(train, weather_train)

del weather_train

gc.collect()
check_missing(train).head(len(train))
categorical_features = []

print(train.columns)

for col in train.columns:

    if train[col].dtype == "object":

        categorical_features.append(col)

print(categorical_features)
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

print(f'columns with unique value in train{one_value_cols}')
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = train["timestamp"].dt.hour

train["day"] = train["timestamp"].dt.day

train["weekend"] = train["timestamp"].dt.weekday

train["month"] = train["timestamp"].dt.month
# # train = pd.get_dummies(train, columns=categorical_features)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(train["primary_use"])

train["primary_use"] = le.transform(train["primary_use"])
drop_columns = ["timestamp", "meter_reading"]

target = np.log1p(train["meter_reading"])

train.drop(drop_columns, axis=1, inplace=True)
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(train, target , test_size= 0.2, random_state=1)
params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'rmse'},

    'learning_rate': 0.3,

    'feature_fraction': 0.8,

    'bagging_fraction': 0.8

}

lgb_train = lgb.Dataset(x_train, y_train)

lgb_test = lgb.Dataset(x_test, y_test)

del x_train, x_test , y_train, y_test
gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=20, verbose_eval = 20)
del lgb_train, lgb_test, train, target

gc.collect()
weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")

print('Size of weather_test_df data', weather_test.shape)

print('Size of test_df data', test.shape)
row_id = test['row_id']

test.drop(['row_id'], axis=1, inplace=True)
for df in [test, weather_test]:

    df = reduce_mem_usage(df)



weather_test.drop(['cloud_coverage', 'precip_depth_1_hr'], axis=1, inplace=True)

weather_test.replace('NaN', np.nan, inplace=True)

for col in ["air_temperature", "dew_temperature", "sea_level_pressure", "wind_direction", "wind_speed"]:

    weather_test[col].fillna(weather_test[col].mean(), inplace=True)



test = merging(test, weather_test)   

del weather_test, building_metadata

gc.collect()

test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = test["timestamp"].dt.hour

test["day"] = test["timestamp"].dt.day

test["weekend"] = test["timestamp"].dt.weekday

test["month"] = test["timestamp"].dt.month

test.drop(["timestamp"], axis=1, inplace=True)



test["primary_use"] = le.transform(test["primary_use"])



del le

gc.collect()
pred = []

step = 50000

for i in range(0, len(test), step):

    pred.extend(np.expm1(gbm.predict(test.iloc[i: min(i+step, len(test)), :], num_iteration=gbm.best_iteration)))
submission = pd.DataFrame({'row_id':row_id, 'meter_reading': pred})

submission['meter_reading'].describe()
submission['meter_reading'] = submission['meter_reading'].apply(lambda x: 0 if x<0 else x)

submission.to_csv("submission.csv", index = False)
submission['meter_reading'].describe()