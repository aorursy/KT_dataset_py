import gc

import os

from pathlib import Path

import random

import sys

import os

from tqdm import tqdm

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import feather

#from IPython.core.display import display, HTML



from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



from sklearn import preprocessing

from sklearn.model_selection import *

import lightgbm as lgb

import xgboost as xgb



import warnings

warnings.filterwarnings('ignore')
%%time

train_df = feather.read_dataframe('../input/ashrae-feather/train.ft')

building_df = feather.read_dataframe('../input/ashrae-feather/building.ft')

weather_train_df = feather.read_dataframe('../input/ashrae-feather/weather_train.ft')

weather_test_df = feather.read_dataframe('../input/ashrae-feather/weather_test.ft')

test_df = feather.read_dataframe('../input/ashrae-feather/test.ft')
weather = pd.concat([weather_train_df,weather_test_df],ignore_index=True)



weather_key = ['site_id', 'timestamp']

full_weather = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
data_to_plot = full_weather.copy()

data_to_plot["hour"] = data_to_plot["timestamp"].dt.hour

count = 1

plt.figure(figsize=(25, 15))

for site_id, data_by_site in data_to_plot.groupby('site_id'):

    by_site_by_hour = data_by_site.groupby('hour').mean()

    ax = plt.subplot(4, 4, count)

    plt.plot(by_site_by_hour.index,by_site_by_hour['air_temperature'],'xb-')

    ax.set_title('site: '+str(site_id))

    count += 1

plt.tight_layout()

plt.show()
# calculate ranks of hourly temperatures within date/site_id chunks

full_weather['temp_rank'] = full_weather.groupby(['site_id', full_weather.timestamp.dt.date])['air_temperature'].rank('average')



# create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)

df_2d = full_weather.groupby(['site_id', full_weather.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)



# Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.

site_ids_offsets = pd.Series(df_2d.values.argmax(axis=1) - 14)

site_ids_offsets.index.name = 'site_id'



def timestamp_align(df):

    df['offset'] = df.site_id.map(site_ids_offsets)

    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))

    df['timestamp'] = df['timestamp_aligned']

    del df['timestamp_aligned']

    return df
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16 or not. feather format does not support float16.

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

#     return df
building_site_dict = dict(zip(building_df['building_id'], building_df['site_id']))

site_meter_raw = train_df[['building_id', 'meter', 'timestamp', 'meter_reading']].copy()

site_meter_raw['site_id'] = site_meter_raw.building_id.map(building_site_dict)

del site_meter_raw['building_id']

site_meter_to_plot = site_meter_raw.copy()

site_meter_to_plot["hour"] = site_meter_to_plot["timestamp"].dt.hour

elec_to_plot = site_meter_to_plot[site_meter_to_plot.meter == 0]
count = 1

plt.figure(figsize=(25, 40))

for site_id, data_by_site in elec_to_plot.groupby('site_id'):

    by_site_by_hour = data_by_site.groupby('hour').mean()

    ax = plt.subplot(15, 4, count)

    plt.plot(by_site_by_hour.index,by_site_by_hour['meter_reading'],'xb-')

    ax.set_title('site: '+str(site_id))

    count += 1

plt.tight_layout()

plt.show()
def preprocess(df):

    df["hour"] = df["timestamp"].dt.hour

    df["weekday"] = df["timestamp"].dt.weekday

    df["month"] = df["timestamp"].dt.month

    

def add_lag_feature(weather_df, window=3):

    group_df = weather_df.groupby('site_id')

    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']

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
%%time

train_df['date'] = train_df['timestamp'].dt.date

train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])



train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20 18")')

train_df = train_df.query('not (building_id == 681 & meter == 0 & timestamp <= "2016-04-27")')

train_df = train_df.query('not (building_id == 761 & meter == 0 & timestamp <= "2016-09-02")')

train_df = train_df.query('not (building_id == 799 & meter == 0 & timestamp <= "2016-09-02")')

train_df = train_df.query('not (building_id == 802 & meter == 0 & timestamp <= "2016-08-24")')

train_df = train_df.query('not (building_id == 1073 & meter == 0 & timestamp <= "2016-10-26")')

train_df = train_df.query('not (building_id == 1094 & meter == 0 & timestamp <= "2016-09-08")')

train_df = train_df.query('not (building_id == 29 & meter == 0 & timestamp <= "2016-08-10")')

train_df = train_df.query('not (building_id == 40 & meter == 0 & timestamp <= "2016-06-04")')

train_df = train_df.query('not (building_id == 45 & meter == 0 & timestamp <= "2016-07")')

train_df = train_df.query('not (building_id == 106 & meter == 0 & timestamp <= "2016-11")')

train_df = train_df.query('not (building_id == 107 & meter == 0 & timestamp >= "2016-11-10")')

train_df = train_df.query('not (building_id == 112 & meter == 0 & timestamp < "2016-10-31 15")')

train_df = train_df.query('not (building_id == 144 & meter == 0 & timestamp > "2016-05-14" & timestamp < "2016-10-31")')

train_df = train_df.query('not (building_id == 147 & meter == 0 & timestamp > "2016-06-05 19" & timestamp < "2016-07-18 15")')

train_df = train_df.query('not (building_id == 171 & meter == 0 & timestamp <= "2016-07-05")')

train_df = train_df.query('not (building_id == 177 & meter == 0 & timestamp > "2016-06-04" & timestamp < "2016-06-25")')

train_df = train_df.query('not (building_id == 258 & meter == 0 & timestamp > "2016-09-26" & timestamp < "2016-12-12")')

train_df = train_df.query('not (building_id == 258 & meter == 0 & timestamp > "2016-08-30" & timestamp < "2016-09-08")')

train_df = train_df.query('not (building_id == 258 & meter == 0 & timestamp > "2016-09-18" & timestamp < "2016-09-25")')

train_df = train_df.query('not (building_id == 260 & meter == 0 & timestamp <= "2016-05-11")')

train_df = train_df.query('not (building_id == 269 & meter == 0 & timestamp > "2016-06-04" & timestamp < "2016-06-25")')

train_df = train_df.query('not (building_id == 304 & meter == 0 & timestamp >= "2016-11-20")')

train_df = train_df.query('not (building_id == 545 & meter == 0 & timestamp > "2016-01-17" & timestamp < "2016-02-10")')

train_df = train_df.query('not (building_id == 604 & meter == 0 & timestamp < "2016-11-21")')

train_df = train_df.query('not (building_id == 693 & meter == 0 & timestamp > "2016-09-07" & timestamp < "2016-11-23")')

train_df = train_df.query('not (building_id == 693 & meter == 0 & timestamp > "2016-07-12" & timestamp < "2016-05-29")')

train_df = train_df.query('not (building_id == 723 & meter == 0 & timestamp > "2016-10-06" & timestamp < "2016-11-22")')

train_df = train_df.query('not (building_id == 733 & meter == 0 & timestamp > "2016-05-29" & timestamp < "2016-06-22")')

train_df = train_df.query('not (building_id == 733 & meter == 0 & timestamp > "2016-05-19" & timestamp < "2016-05-20")')

train_df = train_df.query('not (building_id == 803 & meter == 0 & timestamp > "2016-9-25")')

train_df = train_df.query('not (building_id == 815 & meter == 0 & timestamp > "2016-05-17" & timestamp < "2016-11-17")')

train_df = train_df.query('not (building_id == 848 & meter == 0 & timestamp > "2016-01-15" & timestamp < "2016-03-20")')

train_df = train_df.query('not (building_id == 857 & meter == 0 & timestamp > "2016-04-13")')

train_df = train_df.query('not (building_id == 909 & meter == 0 & timestamp < "2016-02-02")')

train_df = train_df.query('not (building_id == 909 & meter == 0 & timestamp < "2016-06-23")')

train_df = train_df.query('not (building_id == 1008 & meter == 0 & timestamp > "2016-10-30" & timestamp < "2016-11-21")')

train_df = train_df.query('not (building_id == 1113 & meter == 0 & timestamp < "2016-07-27")')

train_df = train_df.query('not (building_id == 1153 & meter == 0 & timestamp < "2016-01-20")')

train_df = train_df.query('not (building_id == 1169 & meter == 0 & timestamp < "2016-08-03")')

train_df = train_df.query('not (building_id == 1170 & meter == 0 & timestamp > "2016-06-30" & timestamp < "2016-07-05")')

train_df = train_df.query('not (building_id == 1221 & meter == 0 & timestamp < "2016-11-04")')

train_df = train_df.query('not (building_id == 1225 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1234 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1233 & building_id <= 1234 & meter == 0 & timestamp > "2016-01-13 22" & timestamp < "2016-03-08 12")')

train_df = train_df.query('not (building_id == 1241 & meter == 0 & timestamp > "2016-07-14" & timestamp < "2016-11-19")')

train_df = train_df.query('not (building_id == 1250 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1255 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1264 & meter == 0 & timestamp > "2016-08-23")')

train_df = train_df.query('not (building_id == 1265 & meter == 0 & timestamp > "2016-05-06" & timestamp < "2016-05-26")')

train_df = train_df.query('not (building_id == 1272 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1275 & building_id <= 1280 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1283 & meter == 0 & timestamp > "2016-07-08" & timestamp < "2016-08-03")')

train_df = train_df.query('not (building_id >= 1291 & building_id <= 1302 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1303 & meter == 0 & timestamp > "2016-07-25 22" & timestamp < "2016-07-27 16")')

train_df = train_df.query('not (building_id == 1303 & meter == 0 & timestamp > "2016-01-26" & timestamp < "2016-06-02 12")')

train_df = train_df.query('not (building_id == 1319 & meter == 0 & timestamp > "2016-05-17 16" & timestamp < "2016-06-07 12")')

train_df = train_df.query('not (building_id == 1319 & meter == 0 & timestamp > "2016-08-18 14" & timestamp < "2016-09-02 14")')

train_df = train_df.query('not (building_id == 1322 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')



# 2nd cleaning

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-10-14 22" & timestamp < "2016-10-17 08")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-07-01 14" & timestamp < "2016-07-05 06")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-10-14 22" & timestamp < "2016-10-17 08")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-07-01 14" & timestamp < "2016-07-05 06")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-10-14 22" & timestamp < "2016-10-17 08")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-07-01 14" & timestamp < "2016-07-05 06")')

train_df = train_df.query('not (building_id == 1272 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1291 & building_id <= 1297 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1300 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1302 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1291 & building_id <= 1299 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1221 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1225 & building_id <= 1226 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1233 & building_id <= 1234 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1241 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1223 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1226 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1233 & building_id <= 1234 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1225 & building_id <= 1226 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1305 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1307 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1223 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1231 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1233 & building_id <= 1234 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1272 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id >= 1275 & building_id <= 1297 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1300 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1302 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1293 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-25 12")')

train_df = train_df.query('not (building_id == 1302 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-25 12")')

train_df = train_df.query('not (building_id == 1223 & meter == 0 & timestamp > "2016-9-28 07" & timestamp < "2016-10-11 18")')

train_df = train_df.query('not (building_id == 1225 & meter == 1 & timestamp > "2016-8-22 23" & timestamp < "2016-10-11 14")')

train_df = train_df.query('not (building_id == 1230 & meter == 1 & timestamp > "2016-8-22 08" & timestamp < "2016-10-05 18")')

train_df = train_df.query('not (building_id == 904 & meter == 0 & timestamp < "2016-02-17 08")')

train_df = train_df.query('not (building_id == 986 & meter == 0 & timestamp < "2016-02-17 08")')

train_df = train_df.query('not (building_id == 954 & meter == 0 & timestamp < "2016-08-08 11")')

train_df = train_df.query('not (building_id == 954 & meter == 0 & timestamp < "2016-06-23 08")')

train_df = train_df.query('not (building_id >= 745 & building_id <= 770 & meter == 1 & timestamp > "2016-10-05 01" & timestamp < "2016-10-10 09")')

train_df = train_df.query('not (building_id >= 774 & building_id <= 787 & meter == 1 & timestamp > "2016-10-05 01" & timestamp < "2016-10-10 09")')



# 3rd cleaning hourly spikes

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-05-11 09" & timestamp < "2016-05-12 01")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-05-11 09" & timestamp < "2016-05-12 01")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-05-11 09" & timestamp < "2016-05-12 01")')



train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp == "2016-02-26 01")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp == "2016-02-26 01")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp == "2016-02-26 01")')



train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-03-29 10" & timestamp < "2016-03-30 12")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-03-29 10" & timestamp < "2016-03-30 12")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-03-29 10" & timestamp < "2016-03-30 12")')



train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-01-19 23" & timestamp < "2016-01-28 15")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-01-19 23" & timestamp < "2016-01-28 15")')

train_df = train_df.query('not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-01-19 23" & timestamp < "2016-01-28 15")')



train_df = train_df.query('not (building_id != 1227 & building_id != 1281 & building_id != 1314 & building_id >=1223 & building_id < 1335 & meter==0 & meter_reading==0)')



# 4th cleaning (some using hindsight from leaks)

train_df = train_df.query('not (building_id >= 1223 & building_id <= 1324 & meter==1 & timestamp > "2016-07-16 04" & timestamp < "2016-07-19 11")')

train_df = train_df.query('not (building_id == 107 & meter == 0 & timestamp <= "2016-07-06")')

train_df = train_df.query('not (building_id == 180 & timestamp >= "2016-02-17 12")')

train_df = train_df.query('not (building_id == 182 & meter == 0)')

train_df = train_df.query('not (building_id == 191 & meter == 0 & timestamp >= "2016-12-22 09")')

train_df = train_df.query('not (building_id == 192 & meter == 1 & timestamp >= "2016-05-09 18")')

train_df = train_df.query('not (building_id == 192 & meter == 3 & timestamp >= "2016-03-29 05" & timestamp <= "2016-04-04 08")')

train_df = train_df.query('not (building_id == 207 & meter == 1 & timestamp > "2016-07-02 20" & timestamp < "2016-08-25 12")')

train_df = train_df.query('not (building_id == 258 & timestamp > "2016-09-18" & timestamp < "2016-12-12 13")')

train_df = train_df.query('not (building_id == 258 & timestamp > "2016-08-29 08" & timestamp < "2016-09-08 14")')

train_df = train_df.query('not (building_id == 257 & meter == 1 & timestamp < "2016-03-25 16")')

train_df = train_df.query('not (building_id == 260 & meter == 1 & timestamp > "2016-05-10 17" & timestamp < "2016-08-17 11")')

train_df = train_df.query('not (building_id == 260 & meter == 1 & timestamp > "2016-08-28 01" & timestamp < "2016-10-31 13")')

train_df = train_df.query('not (building_id == 220 & meter == 1 & timestamp > "2016-09-23 01" & timestamp < "2016-09-23 12")')

train_df = train_df.query('not (building_id == 281 & meter == 1 & timestamp > "2016-10-25 08" & timestamp < "2016-11-04 15")')

train_df = train_df.query('not (building_id == 273 & meter == 1 & timestamp > "2016-04-03 04" & timestamp < "2016-04-29 15")')

train_df = train_df.query('not (building_id == 28 & meter == 0 & timestamp < "2016-10-14 20")')

train_df = train_df.query('not (building_id == 71 & meter == 0 & timestamp < "2016-08-18 20")')

train_df = train_df.query('not (building_id == 76 & meter == 0 & timestamp > "2016-06-04 09" & timestamp < "2016-06-04 14")')

train_df = train_df.query('not (building_id == 101 & meter == 0 & timestamp > "2016-10-12 13" & timestamp < "2016-10-12 18")')

train_df = train_df.query('not (building_id == 7 & meter == 1 & timestamp > "2016-11-03 09" & timestamp < "2016-11-28 14")')

train_df = train_df.query('not (building_id == 9 & meter == 1 & timestamp > "2016-12-06 08")')

train_df = train_df.query('not (building_id == 43 & meter == 1 & timestamp > "2016-04-03 08" & timestamp < "2016-06-06 13")')

train_df = train_df.query('not (building_id == 60 & meter == 1 & timestamp > "2016-05-01 17" & timestamp < "2016-05-01 21")')

train_df = train_df.query('not (building_id == 75 & meter == 1 & timestamp > "2016-08-05 13" & timestamp < "2016-08-26 12")')

train_df = train_df.query('not (building_id == 95 & meter == 1 & timestamp > "2016-08-08 10" & timestamp < "2016-08-26 13")')

train_df = train_df.query('not (building_id == 97 & meter == 1 & timestamp > "2016-08-08 14" & timestamp < "2016-08-25 14")')

train_df = train_df.query('not (building_id == 1232 & meter == 1 & timestamp > "2016-06-23 16" & timestamp < "2016-08-31 20")')

train_df = train_df.query('not (building_id == 1236 & meter == 1 & meter_reading >= 3000)')

train_df = train_df.query('not (building_id == 1239 & meter == 1 & timestamp > "2016-03-11 16" & timestamp < "2016-03-27 17")')

train_df = train_df.query('not (building_id == 1264 & meter == 1 & timestamp > "2016-08-22 17" & timestamp < "2016-09-22 20")')

train_df = train_df.query('not (building_id == 1264 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

train_df = train_df.query('not (building_id == 1269 & meter == 1 & meter_reading >= 2000)')

train_df = train_df.query('not (building_id == 1272 & meter == 1 & timestamp > "2016-08-11 12" & timestamp < "2016-08-30 19")')

train_df = train_df.query('not (building_id == 1273 & meter == 1 & timestamp > "2016-05-31 14" & timestamp < "2016-06-17")')

train_df = train_df.query('not (building_id == 1276 & meter == 1 & timestamp < "2016-02-03 23")')

train_df = train_df.query('not (building_id == 1280 & meter == 1 & timestamp > "2016-05-18" & timestamp < "2016-05-26 09")')

train_df = train_df.query('not (building_id == 1280 & meter == 1 & timestamp > "2016-02-28 23" & timestamp < "2016-05-02 05")')

train_df = train_df.query('not (building_id == 1280 & meter == 1 & timestamp > "2016-06-12 01" & timestamp < "2016-7-07 06")')

train_df = train_df.query('not (building_id == 1288 & meter == 1 & timestamp > "2016-07-07 15" & timestamp < "2016-08-12 17")')

train_df = train_df.query('not (building_id == 1311 & meter == 1 & timestamp > "2016-04-25 18" & timestamp < "2016-05-13 14")')

train_df = train_df.query('not (building_id == 1099 & meter == 2)')



train_df = train_df.query('not (building_id == 1329 & meter == 0 & timestamp > "2016-04-28 00" & timestamp < "2016-04-28 07")')

train_df = train_df.query('not (building_id == 1331 & meter == 0 & timestamp > "2016-04-28 00" & timestamp < "2016-04-28 07")')

train_df = train_df.query('not (building_id == 1427 & meter == 0 & timestamp > "2016-04-11 10" & timestamp < "2016-04-11 14")')

train_df = train_df.query('not (building_id == 1426 & meter == 2 & timestamp > "2016-05-03 09" & timestamp < "2016-05-03 14")')

train_df = train_df.query('not (building_id == 1345 & meter == 0 & timestamp < "2016-03-01")')

train_df = train_df.query('not (building_id == 1346 & timestamp < "2016-03-01")')

train_df = train_df.query('not (building_id == 1359 & meter == 0 & timestamp > "2016-04-25 17" & timestamp < "2016-07-22 14")')

train_df = train_df.query('not (building_id == 1365 & meter == 0 & timestamp > "2016-08-19 00" & timestamp < "2016-08-19 07")')

train_df = train_df.query('not (building_id == 1365 & meter == 0 & timestamp > "2016-06-18 22" & timestamp < "2016-06-19 06")')



train_df = train_df.query('not (building_id == 18 & meter == 0 & timestamp > "2016-06-04 09" & timestamp < "2016-06-04 16")')

train_df = train_df.query('not (building_id == 18 & meter == 0 & timestamp > "2016-11-05 05" & timestamp < "2016-11-05 15")')

train_df = train_df.query('not (building_id == 101 & meter == 0 & meter_reading > 800)')



train_df = train_df.query('not (building_id == 1384 & meter == 0 & meter_reading == 0 )')

train_df = train_df.query('not (building_id >= 1289 & building_id <= 1301 & meter == 2 & meter_reading == 0)')

train_df = train_df.query('not (building_id == 1243 & meter == 2 & meter_reading == 0)')

train_df = train_df.query('not (building_id == 1263 & meter == 2 & meter_reading == 0)')

train_df = train_df.query('not (building_id == 1284 & meter == 2 & meter_reading == 0)')

train_df = train_df.query('not (building_id == 1286 & meter == 2 & meter_reading == 0)')

train_df = train_df.query('not (building_id == 1263 & meter == 0 & timestamp > "2016-11-10 11" & timestamp < "2016-11-10 15")')



train_df = train_df.query('not (building_id == 1238 & meter == 2 & meter_reading == 0)')

train_df = train_df.query('not (building_id == 1329 & meter == 2 & timestamp > "2016-11-21 12" & timestamp < "2016-11-29 12")')

train_df = train_df.query('not (building_id == 1249 & meter == 2 & meter_reading == 0)')



train_df = train_df.query('not (building_id == 1250 & meter == 2 & meter_reading == 0)')

train_df = train_df.query('not (building_id == 1256 & meter == 2 & timestamp > "2016-03-05 18" & timestamp < "2016-03-05 22")')

train_df = train_df.query('not (building_id == 1256 & meter == 2 & timestamp > "2016-03-27 00" & timestamp < "2016-03-27 23")')

train_df = train_df.query('not (building_id == 1256 & meter == 2 & timestamp > "2016-04-11 09" & timestamp < "2016-04-13 03")')

train_df = train_df.query('not (building_id == 1256 & meter == 2 & timestamp > "2016-04-29 00" & timestamp < "2016-04-30 15")')

train_df = train_df.query('not (building_id == 1303 & meter == 2 & timestamp < "2016-06-06 19")')

train_df = train_df.query('not (building_id >= 1223 & building_id <= 1324 & meter == 1 & timestamp > "2016-08-11 17" & timestamp < "2016-08-12 17")')

train_df = train_df.query('not (building_id >= 1223 & building_id <= 1324 & building_id != 1296 & building_id != 129 & building_id != 1298 & building_id != 1299 & meter == 2 & timestamp > "2016-08-11 17" & timestamp < "2016-08-12 17")')

train_df = train_df.query('not (building_id >= 1223 & building_id <= 1324 & meter == 3 & timestamp > "2016-08-11 17" & timestamp < "2016-08-12 17")')
train_df.head()
debug = False    

preprocess(train_df)



bm_cols = ['building_id', 'weekday', 'hour',]

bm = train_df.groupby(bm_cols)['meter_reading'].mean().rename('b_week_hour').to_frame()

train_df = train_df.merge(bm, right_index=True, left_on=bm_cols, how='left')



# https://www.kaggle.com/ryches/simple-lgbm-solution

df_group = train_df.groupby('building_id')['meter_reading_log1p']

building_mean = df_group.mean().astype(np.float16)

building_median = df_group.median().astype(np.float16)

building_min = df_group.min().astype(np.float16)

building_max = df_group.max().astype(np.float16)

building_std = df_group.std().astype(np.float16)



train_df['building_mean'] = train_df['building_id'].map(building_mean)

train_df['building_median'] = train_df['building_id'].map(building_median)

train_df['building_min'] = train_df['building_id'].map(building_min)

train_df['building_max'] = train_df['building_id'].map(building_max)

train_df['building_std'] = train_df['building_id'].map(building_std)



weather_train_df = timestamp_align(weather_train_df)

weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))



# add_lag_feature(weather_train_df, window=3)

add_lag_feature(weather_train_df, window=72)



primary_use_list = building_df['primary_use'].unique()

primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 

building_df['primary_use'] = building_df['primary_use'].map(primary_use_dict)



gc.collect()



reduce_mem_usage(train_df, use_float16=True)

reduce_mem_usage(building_df, use_float16=True)

reduce_mem_usage(weather_train_df, use_float16=True)
category_cols = ['building_id', 'primary_use', ]  # , 'meter'

feature_cols = ['square_feet', 'year_built'] + [#'sin_hour', 'cos_hour', 

    'hour','weekday', # 'month'

    'building_median'] + ['b_week_hour',

    'air_temperature', 'cloud_coverage',

    'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

    'wind_direction', 'wind_speed', 'air_temperature_mean_lag72',

    'air_temperature_max_lag72', 'air_temperature_min_lag72',

    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',

    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',

    'sea_level_pressure_mean_lag72', 'wind_direction_mean_lag72',

    'wind_speed_mean_lag72',]# 'air_temperature_mean_lag3',

#     'air_temperature_max_lag3',

#     'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',

#     'dew_temperature_mean_lag3',

#     'precip_depth_1_hr_mean_lag3', 'sea_level_pressure_mean_lag3',

#     'wind_direction_mean_lag3', 'wind_speed_mean_lag3']
def create_X_y(train_df, target_meter, use_month=False):

    target_train_df = train_df[train_df['meter'] == target_meter]

    target_train_df = target_train_df.merge(building_df, on='building_id', how='left')

    target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    if not use_month:

        X_train = target_train_df[feature_cols + category_cols]

    else:

        X_train = target_train_df[feature_cols + category_cols + ['month']]

    y_train = target_train_df['meter_reading_log1p'].values



    del target_train_df

    return X_train, y_train



def fit_lgbm(train, val, devices=(-1,), seed=None, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):

    """Train Light GBM model"""

    X_train, y_train = train

    X_valid, y_valid = val

    metric = 'l2'

    params = {'num_leaves': 100,

              'objective': 'regression',

#               'max_depth': -1,

              'learning_rate': lr,

              "boosting": "gbdt",

              "bagging_freq": 6,

              "bagging_fraction": bf,

              "feature_fraction": 0.9,

              "metric": metric,

              "num_threads": 5,

#               "verbosity": -1,

#               'reg_alpha': 0.1,

#               'reg_lambda': 0.3

              }

    device = devices[0]

    if device == -1:

        # use cpu

        pass

    else:

        # use gpu

        print(f'using gpu device_id {device}...')

        params.update({'device': 'gpu', 'gpu_device_id': device})



    params['seed'] = seed



    early_stop = 20

    verbose_eval = 100



    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)

    watchlist = [d_train, d_valid]



    model = lgb.train(params,

                      train_set=d_train,

                      num_boost_round=num_rounds,

                      valid_sets=watchlist,

                      verbose_eval=verbose_eval,

                      early_stopping_rounds=early_stop)



    # predictions

    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    

    log = {'train/mae': model.best_score['training']['l2'],

           'valid/mae': model.best_score['valid_1']['l2']}

    return model, y_pred_valid, log



folds = 5

seed = 666

shuffle = False

kf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=seed)
%%time

target_meter = 0

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])

gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models0 = []

for train_idx, valid_idx in kf.split(X_train, X_train['building_id']):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,

                                        num_rounds=1000, lr=0.05, bf=0.9)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models0.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

del X_train, y_train

gc.collect()
%%time

target_meter = 1

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])

gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models1 = []

for train_idx, valid_idx in kf.split(X_train, X_train['building_id']):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=1000,

                                       lr=0.05, bf=0.9)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models1.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

del X_train, y_train

gc.collect()
%%time

target_meter = 2

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])



gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models2 = []

for train_idx, valid_idx in kf.split(X_train, X_train['building_id']):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,

                                        num_rounds=1000, lr=0.05, bf=0.9)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models2.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

del X_train, y_train

gc.collect()
%%time

target_meter = 3

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])



gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models3 = []

for train_idx, valid_idx in kf.split(X_train, X_train['building_id']):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=1000,

                                       lr=0.03, bf=0.9)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models3.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

del X_train, y_train

gc.collect()
print('preprocessing building...')

test_df['date'] = test_df['timestamp'].dt.date

preprocess(test_df)

test_df['building_mean'] = test_df['building_id'].map(building_mean)

test_df['building_median'] = test_df['building_id'].map(building_median)

test_df['building_min'] = test_df['building_id'].map(building_min)

test_df['building_max'] = test_df['building_id'].map(building_max)

test_df['building_std'] = test_df['building_id'].map(building_std)



print('preprocessing weather...')

weather_test_df = timestamp_align(weather_test_df)

weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())



# add_lag_feature(weather_test_df, window=3)

add_lag_feature(weather_test_df, window=72)



print('reduce mem usage...')

reduce_mem_usage(test_df, use_float16=True)

reduce_mem_usage(weather_test_df, use_float16=True)



gc.collect()
sub = pd.read_feather('../input/ashrae-feather/sample_submission.ft')
def create_X(test_df, target_meter):

    target_test_df = test_df[test_df['meter'] == target_meter]

    target_test_df = target_test_df.merge(building_df, on='building_id', how='left')

    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

    preprocess(target_test_df)

    target_test_df = target_test_df.merge(bm, right_index=True, left_on=bm_cols, how='left')

    

    X_test = target_test_df[feature_cols + category_cols]

    return X_test
def pred(X_test, models, batch_size=2000000):

    iterations = (X_test.shape[0] + batch_size -1) // batch_size



    y_test_pred_total = np.zeros(X_test.shape[0])

    for i, model in enumerate(models):

        for k in tqdm(range(iterations)):

            y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)

            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test



    y_test_pred_total /= len(models)

    return y_test_pred_total
%%time

X_test = create_X(test_df, target_meter=0)



y_test0 = pred(X_test, models0)



sns.distplot(y_test0)



del X_test

gc.collect()
%%time

X_test = create_X(test_df, target_meter=1)



y_test1 = pred(X_test, models1)

sns.distplot(y_test1)



del X_test

gc.collect()
%%time

X_test = create_X(test_df, target_meter=2)



y_test2 = pred(X_test, models2)

sns.distplot(y_test2)



del X_test

gc.collect()
%%time

X_test = create_X(test_df, target_meter=3)



y_test3 = pred(X_test, models3)

sns.distplot(y_test3)



del X_test

gc.collect()
sub.loc[test_df['meter'] == 0, 'meter_reading'] = np.expm1(y_test0)

sub.loc[test_df['meter'] == 1, 'meter_reading'] = np.expm1(y_test1)

sub.loc[test_df['meter'] == 2, 'meter_reading'] = np.expm1(y_test2)

sub.loc[test_df['meter'] == 3, 'meter_reading'] = np.expm1(y_test3)
%%time

sub['meter_reading'] = np.clip(sub['meter_reading'].values, a_min=0, a_max=None)

sub['meter_reading'] = sub['meter_reading'].astype('float32')

sub.to_csv('submission.csv', index=False, chunksize=25000, float_format='%.4f')

sub.head()
np.log1p(sub['meter_reading']).hist()

plt.show()
plt.yscale('log')

sub['meter_reading'].hist()

plt.show()
def plot_feature_importance(model):

    importance_df = pd.DataFrame(model.feature_importance(),

                                 index=feature_cols + category_cols,

                                 columns=['importance']).sort_values('importance')

    fig, ax = plt.subplots(figsize=(8, 8))

    importance_df.plot.barh(ax=ax)

    plt.show()
plot_feature_importance(models0[1])
plot_feature_importance(models1[1])
plot_feature_importance(models2[1])
plot_feature_importance(models3[1])
def rmse(ytrue, ypred):

    return np.sqrt(np.mean(np.square(ypred - ytrue), axis=0))

def rmsle(ytrue, ypred):

    return np.sqrt(np.mean(np.square(np.log1p(ypred) - np.log1p(ytrue)), axis=0))
sub.head()
print(f"sub mean: {sub['meter_reading'].mean():.4f}")

print(f"sub std: {sub['meter_reading'].std():.4f}")

print(f"sub min: {sub['meter_reading'].min():.4f}")

print(f"sub max: {sub['meter_reading'].max():.4f}")
sns.distplot(np.log1p(sub['meter_reading'].values), kde=False);
site0 = pd.read_feather('../input/ucf-building-meter-reading/site0.ft')

df_test = pd.read_feather('../input/ashrae-feather/test.ft')
merged = df_test.merge(site0, left_on=['building_id', 'meter', 'timestamp'], 

              right_on=['building_id', 'meter', 'timestamp'], how='left')
ytrue = merged[~merged['meter_reading'].isna()]['meter_reading']

pred = sub[~merged['meter_reading'].isna()]['meter_reading']
del site0, merged

print(f'RMSLE of buildings 0-104: {rmsle(ytrue, pred):.4f}')
site1 = pd.read_feather('../input/ucl-data-leakage-episode-2/site1.ft')

site1 = site1.query('timestamp >= 2017')
merged = df_test.merge(site1, left_on=['building_id', 'meter', 'timestamp'], 

              right_on=['building_id', 'meter', 'timestamp'], how='left')
ytrue = merged[~merged['meter_reading'].isna()]['meter_reading']

pred = sub[~merged['meter_reading'].isna()]['meter_reading']
del merged, site1

print(f'RMSLE of buildings 105-155: {rmsle(ytrue, pred):.4f}')
site2 = pd.read_feather('../input/asu-feather/site2.ft')

site2 = site2.query('timestamp >= 2017')
merged = df_test.merge(site2, left_on=['building_id', 'meter', 'timestamp'], 

              right_on=['building_id', 'meter', 'timestamp'], how='left')
ytrue = merged[~merged['meter_reading'].isna()]['meter_reading']

pred = sub[~merged['meter_reading'].isna()]['meter_reading']
del site2, merged

print(f'RMSLE of buildings 156-290: {rmsle(ytrue, pred):.4f}')
site4 = pd.read_feather('../input/ucb-feather/site4.ft')

site4 = site4.query('timestamp >= 2017')
merged = df_test.merge(site4, left_on=['building_id', 'timestamp'], 

              right_on=['building_id', 'timestamp'], how='left')
ytrue = merged[~merged['meter_reading'].isna()]['meter_reading']

pred = sub[~merged['meter_reading'].isna()]['meter_reading']
del site4, merged

print(f'RMSLE of 74/91 buildings : {rmsle(ytrue, pred):.4f}')
site15 = pd.read_feather('../input/cornell-feather/site15.ft')

site15 = site15.query('timestamp >= 2017')

site15 = site15.drop_duplicates()
merged = df_test.merge(site15, left_on=['building_id', 'meter', 'timestamp'], 

              right_on=['building_id', 'meter', 'timestamp'], how='left')
ytrue = merged[~merged['meter_reading'].isna()]['meter_reading']

pred = sub[~merged['meter_reading'].isna()]['meter_reading']
del site15, merged

print(f'RMSLE of buildings 1325-1448: {rmsle(ytrue, pred):.4f}')
site012 = pd.read_feather('../input/comb-leaked-dataset/site012.ft')

site012 = site012.query('timestamp >= 2017')
merged = df_test.merge(site012, left_on=['building_id', 'meter', 'timestamp'], 

              right_on=['building_id', 'meter', 'timestamp'], how='left')
ytrue = merged[~merged['meter_reading'].isna()]['meter_reading']

pred = sub[~merged['meter_reading'].isna()]['meter_reading']
del site012, merged

print(f'RMSLE of buildings 0-290: {rmsle(ytrue, pred):.4f}')