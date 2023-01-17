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
import gc

gc.collect()

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from dask import dataframe as dd



import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder



sns.set()

%matplotlib inline
#Some functions to fill data, add features and reduce memory usage



def fill_weather_dataset(weather_df):

    

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]



    missing_hours = []

    for site_id in range(16):

        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])

        new_rows['site_id'] = site_id

        weather_df = pd.concat([weather_df,new_rows])



        weather_df = weather_df.reset_index(drop=True)           



    # Add new Features

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])

    weather_df["day"] = weather_df["datetime"].dt.day

    weather_df["week"] = weather_df["datetime"].dt.week

    weather_df["month"] = weather_df["datetime"].dt.month

    

    # Reset Index for Fast Update

    weather_df = weather_df.set_index(['site_id','day','month'])



    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    weather_df.update(air_temperature_filler,overwrite=False)



    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])



    weather_df.update(cloud_coverage_filler,overwrite=False)



    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    weather_df.update(due_temperature_filler,overwrite=False)



    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    weather_df.update(wind_direction_filler,overwrite=False)



    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    weather_df.update(wind_speed_filler,overwrite=False)



    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])



    weather_df.update(precip_depth_filler,overwrite=False)



    weather_df = weather_df.reset_index()

    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)

        

    return weather_df
def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if isinstance(df[col], datetime.datetime) or pd.api.types.is_categorical_dtype(df[col]):

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
def add_lag_feature(weather_df, window=3):

    group_df = weather_df.groupby('site_id')

    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 

            'sea_level_pressure']

    rolled = group_df[cols].rolling(window=window, min_periods=0)

    lag_mean = rolled.mean().reset_index().astype(np.float16)

    lag_max = rolled.max().reset_index().astype(np.float16)

    lag_min = rolled.min().reset_index().astype(np.float16)

    lag_std = rolled.std().reset_index().astype(np.float16)

    for col in cols:

        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]

        weather_df[f'{col}_max_lag{window}'] = lag_max[col]

        weather_df[f'{col}_min_lag{window}'] = lag_min[col]

        weather_df[f'{col}_std_lag{window}'] = lag_std[col]
def addFeatures(df):

    

    # Sort by timestamp

    df.sort_values("timestamp")

    df.reset_index(drop=True)

    

    # Add more features

    df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")

    df["hour"] = df["timestamp"].dt.hour

    df["weekend"] = df["timestamp"].dt.weekday

    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                    "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                    "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                    "2019-01-01"]

    df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)

    #df['square_feet'] =  np.log1p(df['square_feet'])

    

    building_mean = df_group.mean().astype(np.float16)

    building_median = df_group.median().astype(np.float16)

    building_min = df_group.min().astype(np.float16)

    building_max = df_group.max().astype(np.float16)

    building_std = df_group.std().astype(np.float16)



    df['building_mean'] = df['building_id'].map(building_mean)

    df['building_median'] = df['building_id'].map(building_median)

    df['building_min'] = df['building_id'].map(building_min)

    df['building_max'] = df['building_id'].map(building_max)

    df['building_std'] = df['building_id'].map(building_std)

    

    # Remove Unused Columns

    drop = [#"timestamp","sea_level_pressure", "wind_direction", "wind_speed",

            "year_built","floor_count",'timestamp']

    df = df.drop(drop, axis=1)

    gc.collect()

    

    # Encode Categorical Data

    #le = LabelEncoder()

    #df["primary_use"] = le.fit_transform(df["primary_use"])

    

    return df
building_metadata = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

building_metadata['primary_use'] = building_metadata['primary_use'].astype('category')

le = LabelEncoder()

building_metadata["primary_use"] = le.fit_transform(building_metadata["primary_use"])

building_metadata['square_feet'] =  np.log1p(building_metadata['square_feet'])

building_metadata = reduce_mem_usage(building_metadata,use_float16=True)
print('train data:')

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

# Remove outliers

#train = train[ train['building_id'] != 1099 ]

train = train[~((train['meter'] == 2) & (train['building_id'] == 1099))]

train = train.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

train['meter_reading_log1p'] = np.log1p(train['meter_reading'])

df_group = train.groupby('building_id')['meter_reading_log1p']



#Get part of data with full set of timestamps

count_full = train.groupby('building_id')['timestamp'].nunique()

#Remember count_full is a Series object

count_full = count_full[count_full==count_full.max()]

#ids with whole length

print(count_full.index)

train = train[train['building_id'].isin(count_full.index)]
#Fill Weather Information

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_train = fill_weather_dataset(weather_train)

#weather_train = weather_train.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

add_lag_feature(weather_train, window=3)

add_lag_feature(weather_train, window=72)
#Memory reduction

train = reduce_mem_usage(train,use_float16=True)

weather_train = reduce_mem_usage(weather_train,use_float16=True)
#Merge data

train = train.merge(building_metadata, on='building_id', how='left')

train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')

del weather_train

gc.collect()
#Add features

train = addFeatures(train)

train.tail()
#Get features and target variables

def get_train_data(df, site_id):

    df_ = df[df['meter']==mtype]

    target = df_["meter_reading_log1p"]

    features = df_.drop(['meter_reading','meter_reading_log1p'], axis = 1)

    return features, target
categorical_features = ["building_id", "site_id", "meter", "is_holiday", "weekend", 'primary_use']



params = {

    "objective": "regression",

    "boosting": "gbdt",

    "num_leaves": 31,

    "learning_rate": 0.05,

    "feature_fraction": 0.85,

    "reg_lambda": 2,

    "metric": "rmse",

}



kf = KFold(n_splits=3)

models = []

for mtype in [0,1,2,3]:

    print(f'training meter: {mtype}')

    features, target = get_train_data(train, mtype); tmp = []

    for train_index,test_index in kf.split(features, target):

        train_features = features.iloc[train_index]

        train_target = target.iloc[train_index]

    

        test_features = features.iloc[test_index]

        test_target = target.iloc[test_index]

    

        d_training = lgb.Dataset(train_features, label=train_target,

                                 categorical_feature=categorical_features, free_raw_data=False)

        d_test = lgb.Dataset(test_features, label=test_target,

                             categorical_feature=categorical_features, free_raw_data=False)

    

        model = lgb.train(params, train_set=d_training, num_boost_round=1000, 

                          valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)

        tmp.append(model)

    models.append(tmp)



del train_features, train_target, test_features, test_target, d_training, d_test, features, target, train

gc.collect()
#Load test data

test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

row_ids = test["row_id"]

#ref = test[['row_id','meter']]

test = test.drop("row_id", axis=1)

test = reduce_mem_usage(test)

#td = dd.from_pandas(test, npartitions=20)

#del test

gc.collect()
weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

weather_test = fill_weather_dataset(weather_test)

#weather_test = weather_test.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

weather_test = reduce_mem_usage(weather_test)

add_lag_feature(weather_test, window=3)

add_lag_feature(weather_test, window=72)
sub = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv', 

                  dtype={'row_id':np.uint16, 'meter_reading':np.float32})

sub['row_id'] = row_ids

sub
def pred(df, models):

    yp_total = np.zeros(df.shape[0])

    for i, model in enumerate(models):

        print(f'predicting model-{i}')

        yp = model.predict(df, num_iteration=model.best_iteration)

        yp_total += yp



    yp_total /= len(models)

    return yp_total
n = 1000000

for mtype in [0,1,2,3]:

    tst = test.loc[test['meter']==mtype]

    tst = tst.merge(building_metadata, on='building_id', how='left')

    tst = tst.merge(weather_test, on=['site_id', 'timestamp'], how='left')

    tst = addFeatures(tst)

    print(f'meter-{mtype} dataframe shape is {tst.shape}')

    #print(tst.columns)

    gen = (tst[i:i+n] for i in range(0,tst.shape[0],n))

    p_full = []

    for x in gen:

        p = pred(x, models[mtype])

        p_full.append(p)

    p_full = np.concatenate(p_full)

    print(f'predicted array has shape {p_full.shape}')

    sub.loc[test['meter']==mtype, 'meter_reading'] = np.expm1(p_full)

    del tst

    gc.collect()
sub
sub.to_csv("submission.csv", index=False, float_format='%.5f')