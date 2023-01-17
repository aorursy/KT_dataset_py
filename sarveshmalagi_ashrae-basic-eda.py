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
train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
def reduce_mem_usage(df):



    for col in df.columns:

        col_type = df[col].dtype



        if col_type != object:

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

#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

#                     df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float32)



    return df
train = reduce_mem_usage(train)

weather_train = reduce_mem_usage(weather_train)

building_metadata = reduce_mem_usage(building_metadata)
import gc

gc.collect()
train.info()
def calculate_iqr(col):

    Q1 = col.quantile(0.25)

    Q3 = col.quantile(0.75)

    IQR = Q3 - Q1

    return IQR, Q1, Q3
def feature_engineering(df, weather_data, building_metadata, is_train=False):

    

    #Extract datetime related features

    df['timestamp'] = pd.to_datetime(df.timestamp)

    weather_data['timestamp'] = pd.to_datetime(weather_data.timestamp)

    df['hour'] = df.timestamp.dt.hour

    df['day'] = df.timestamp.dt.day

    df['month'] = df.timestamp.dt.month

    df['quarter'] = df.timestamp.dt.quarter

    

    #Merge with building_metadata and weather data

    df = df.merge(building_metadata, left_on='building_id', right_on='building_id')

    df = df.merge(weather_data, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'])

    

    #eliminate outliers

    if is_train:

        IQR, Q1, Q3 = calculate_iqr(df['meter_reading'])

        df = df[~((df['meter_reading'] < (Q1 - 1.5 * IQR)) |(df['meter_reading'] > (Q3 + 1.5 * IQR)))]

        df['meter_reading'] = np.log1p(df.meter_reading)

    

#     del df['timestamp']

    

    #OHE

    categorical_columns = ['month', 'meter', 'primary_use']

    for col in categorical_columns:

        df = pd.concat([df, pd.get_dummies(df[col]).rename(columns=lambda x: col + '_' + str(x))], axis=1, sort=False)

#         del df[col]

    

    return df
train_fe = feature_engineering(train, weather_train, building_metadata, True)
import seaborn as sns

import matplotlib.pyplot as plt
timestamp_grouped_data = train_fe.groupby('month')['meter_reading'].mean().reset_index()

sns.lineplot(x='month', y='meter_reading', data=timestamp_grouped_data)
hourly_grouped_data = train_fe.groupby('hour')['meter_reading'].mean().reset_index()

sns.lineplot(x='hour', y='meter_reading', data=hourly_grouped_data)
primary_use_grouped_hourly = train_fe.groupby(['hour', 'primary_use'])['meter_reading'].mean().reset_index()

hourly_primary_use_grid = sns.FacetGrid(primary_use_grouped_hourly, row='primary_use', height=2, aspect=3)

hourly_primary_use_grid.map(sns.lineplot, 'hour', 'meter_reading')
primary_use_grouped_monthly = train_fe.groupby(['month', 'primary_use'])['meter_reading'].mean().reset_index()

monthly_primary_use_grid = sns.FacetGrid(primary_use_grouped_monthly, row='primary_use', height=2, aspect=3)

monthly_primary_use_grid.map(sns.lineplot, 'month', 'meter_reading')
meter_type_grouped = train_fe.groupby(['meter'])['meter_reading'].sum().reset_index()

sns.catplot(x='meter', y='meter_reading', data=meter_type_grouped, kind="bar")
meter_grouped_monthly = train_fe.groupby(['meter', 'month'])['meter_reading'].mean().reset_index()

meter_type_month_grouped_grid = sns.FacetGrid(meter_grouped_monthly, row='meter', height=2, aspect=3)

meter_type_month_grouped_grid.map(sns.lineplot, 'month', 'meter_reading')
Y = train_fe['meter_reading']



cols_to_delete = ['month', 'meter', 'primary_use', 'meter_reading', 'timestamp']

for col in cols_to_delete:

    del train_fe[col]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_fe, Y, test_size=0.3, random_state=101)
import lightgbm as lgb
params = {

        "objective" : "regression", "metric" : "rmse", "max_depth" : 5,

        "num_leaves" : 50, "learning_rate" : 0.01, "bagging_fraction" : 0.9,

        "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.8

    }
record = dict()



model = lgb.train(params

                      , lgb.Dataset(X_train, Y_train)

                      , num_boost_round = 100

                      , valid_sets = [lgb.Dataset(X_test, Y_test)]

                      , verbose_eval = True

                      , early_stopping_rounds = 20

                      , callbacks = [lgb.record_evaluation(record)]

                     )
lgb.plot_importance(model, importance_type='split', max_num_features=20)