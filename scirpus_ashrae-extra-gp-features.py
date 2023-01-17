import gc

import os

from pathlib import Path

import random

import sys



from tqdm import tqdm_notebook as tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold



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

    

    return df
!ls ../input
%%time

root = Path('../input/ashrae-feather-format-for-fast-loading')



train_df = pd.read_feather(root/'train.feather')

weather_train_df = pd.read_feather(root/'weather_train.feather')

building_meta_df = pd.read_feather(root/'building_metadata.feather')
train_df['date'] = train_df['timestamp'].dt.date

train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
def plot_date_usage(train_df, meter=0, building_id=0):

    train_temp_df = train_df[train_df['meter'] == meter]

    train_temp_df = train_temp_df[train_temp_df['building_id'] == building_id]    

    train_temp_df_meter = train_temp_df.groupby('date')['meter_reading_log1p'].sum()

    train_temp_df_meter = train_temp_df_meter.to_frame().reset_index()

    fig = px.line(train_temp_df_meter, x='date', y='meter_reading_log1p')

    fig.show()
plot_date_usage(train_df, meter=0, building_id=0)
building_meta_df[building_meta_df.site_id == 0]
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
debug = False
def preprocess(df):

    df["hour"] = df["timestamp"].dt.hour

#     df["day"] = df["timestamp"].dt.day

    df["weekend"] = df["timestamp"].dt.weekday

    df["month"] = df["timestamp"].dt.month

    df["dayofweek"] = df["timestamp"].dt.dayofweek



#     hour_rad = df["hour"].values / 24. * 2 * np.pi

#     df["hour_sin"] = np.sin(hour_rad)

#     df["hour_cos"] = np.cos(hour_rad)
preprocess(train_df)
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
building_mean.head()
weather_train_df.head()
# weather_train_df.describe()
weather_train_df.isna().sum()
weather_train_df.shape
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
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
add_lag_feature(weather_train_df, window=3)

add_lag_feature(weather_train_df, window=72)
weather_train_df.head()
weather_train_df.columns
# categorize primary_use column to reduce memory on merge...



primary_use_list = building_meta_df['primary_use'].unique()

primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 

print('primary_use_dict: ', primary_use_dict)

building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)



gc.collect()
reduce_mem_usage(train_df, use_float16=True)

reduce_mem_usage(building_meta_df, use_float16=True)

reduce_mem_usage(weather_train_df, use_float16=True)
building_meta_df.head()
category_cols = ['building_id', 'site_id', 'primary_use']  # , 'meter'

feature_cols = ['square_feet', 'year_built'] + [

    'hour', 'weekend', # 'month' , 'dayofweek'

    'building_median'] + [

    'air_temperature', 'cloud_coverage',

    'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

    'wind_direction', 'wind_speed', 'air_temperature_mean_lag72',

    'air_temperature_max_lag72', 'air_temperature_min_lag72',

    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',

    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',

    'sea_level_pressure_mean_lag72', 'wind_direction_mean_lag72',

    'wind_speed_mean_lag72', 'air_temperature_mean_lag3',

    'air_temperature_max_lag3',

    'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',

    'dew_temperature_mean_lag3',

    'precip_depth_1_hr_mean_lag3', 'sea_level_pressure_mean_lag3',

    'wind_direction_mean_lag3', 'wind_speed_mean_lag3']
def create_X_y(train_df, target_meter):

    target_train_df = train_df[train_df['meter'] == target_meter]

    target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')

    target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    X_train = target_train_df[feature_cols + category_cols]

    y_train = target_train_df['meter_reading_log1p'].values



    del target_train_df

    return X_train, y_train
def GPMeter0(data):

    return (4.093981 +

            1.0*np.tanh(((data["building_median"]) - ((((3.47548198699951172)) + (np.tanh(((((((3.47548198699951172)) + ((-1.0*((data["building_median"])))))) + (((np.where(data["cloud_coverage_mean_lag72"] < -9998, (-1.0*((data["sea_level_pressure"]))), np.tanh(((3.47548198699951172))) )) * 2.0)))))))))) +

            1.0*np.tanh((((((-1.0*((((np.tanh((((((data["building_median"]) + (((-2.0) * 2.0)))) + (((((data["building_median"]) + (((-2.0) * 2.0)))) / 2.0)))))) / 2.0))))) + (((data["building_median"]) + (((-2.0) * 2.0)))))) / 2.0)) +

            0.964338*np.tanh(np.where(np.where(data["wind_speed_mean_lag72"] < -9998, ((-1.0) / 2.0), ((((data["cloud_coverage"]) * 2.0)) * 2.0) ) < -9998, np.where(np.where(data["wind_speed_mean_lag72"] < -9998, 0.0, ((((data["year_built"]) + (((data["hour"]) - (data["weekend"]))))) * 2.0) ) < -9998, 0.0, ((np.tanh((1.0))) * 2.0) ), 0.0 )) +

            1.0*np.tanh((-1.0*(((((((2.0) + ((-1.0*(((((np.tanh(((-1.0*((((((np.where(np.tanh((data["air_temperature_std_lag72"])) < -9998, data["building_median"], data["site_id"] )) - (((((data["building_median"]) / 2.0)) * 2.0)))) - (2.0)))))))) + (data["building_median"]))/2.0))))))/2.0)) / 2.0))))) +

            0.871519*np.tanh(((np.tanh((((0.0) - (((np.tanh((((1.0) - (np.tanh((((((data["hour"]) / 2.0)) / 2.0)))))))) / 2.0)))))) - (((((1.0) - (np.tanh((data["building_median"]))))) * 2.0)))) +

            1.0*np.tanh(((((((((((1.0) - (np.tanh(((((np.where(data["precip_depth_1_hr_mean_lag3"] > -9998, data["air_temperature_mean_lag3"], (((-1.0*((data["cloud_coverage_mean_lag3"])))) / 2.0) )) + ((-1.0*((data["cloud_coverage"])))))/2.0)))))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(((((np.where(data["air_temperature_mean_lag3"] > -9998, np.tanh((np.where(((data["sea_level_pressure_mean_lag72"]) + (((data["weekend"]) / 2.0))) > -9998, np.where(data["sea_level_pressure_mean_lag72"] > -9998, (0.0), data["sea_level_pressure_mean_lag72"] ), np.tanh((data["building_median"])) ))), np.tanh((data["building_median"])) )) / 2.0)) / 2.0)) +

            0.0*np.tanh(0.0) +

            0.929653*np.tanh((((-1.0) + (np.tanh(((((((((np.where(((data["air_temperature_mean_lag3"]) * (((0.0) + (((((data["cloud_coverage_mean_lag72"]) + (data["cloud_coverage_mean_lag72"]))) * 2.0))))) > -9998, data["site_id"], data["dew_temperature_mean_lag72"] )) + (np.tanh((data["cloud_coverage_mean_lag72"]))))) * 2.0)) + (data["site_id"]))/2.0)))))/2.0)) +

            0.0*np.tanh(0.0))



def GPMeter1(data):

    return (4.246485 +

            1.0*np.tanh((((data["building_median"]) + ((((-3.0) + (((((((((data["building_median"]) - ((6.0)))) + (data["air_temperature_mean_lag72"]))) - ((((6.0)) * (3.0))))) + (((data["building_median"]) - ((((((data["wind_speed_mean_lag72"]) - (data["building_median"]))) + (((3.0) - ((-1.0*((data["air_temperature_mean_lag72"])))))))/2.0)))))))/2.0)))/2.0)) +

            1.0*np.tanh(((((np.where(data["cloud_coverage"] > -9998, ((np.where(data["wind_speed"] > -9998, ((((np.where(data["wind_direction"] > -9998, data["dew_temperature"], data["wind_speed"] )) / 2.0)) - (data["cloud_coverage"])), data["precip_depth_1_hr_mean_lag72"] )) / 2.0), ((data["air_temperature_min_lag3"]) - ((((((data["wind_speed"]) + ((8.0)))/2.0)) + ((8.0))))) )) / 2.0)) / 2.0)) +

            1.0*np.tanh((((((data["building_median"]) - ((4.47872495651245117)))) + (np.tanh((((0.0) + (((((((data["building_median"]) * (((np.where(data["wind_direction_mean_lag3"] < -9998, data["building_id"], data["building_median"] )) - (np.where(data["building_id"] < -9998, -1.0, (4.47872495651245117) )))))) - (data["site_id"]))) / 2.0)))))))/2.0)) +

            1.0*np.tanh(((((np.tanh((np.where(data["air_temperature_mean_lag72"] > -9998, np.where(data["cloud_coverage"] > -9998, (((data["hour"]) + ((-1.0*(((((((2.41199660301208496)) * 2.0)) * 2.0))))))/2.0), (-1.0*((((((data["wind_speed_mean_lag3"]) * 2.0)) * 2.0)))) ), data["weekend"] )))) / 2.0)) / 2.0)) +

            1.0*np.tanh(((np.where((((-1.0) + (-1.0))/2.0) < -9998, data["air_temperature_min_lag72"], (((np.tanh((np.tanh((-1.0))))) + (np.where(((data["air_temperature_min_lag72"]) * 2.0) < -9998, data["primary_use"], np.tanh((((((((((data["primary_use"]) + (data["air_temperature_min_lag72"]))/2.0)) + ((0.27198320627212524)))/2.0)) + ((((-1.0) + (-1.0))/2.0))))) )))/2.0) )) / 2.0)) +

            0.918417*np.tanh((((np.tanh(((((7.0)) - ((((((7.0)) - (((data["building_median"]) - (((data["cloud_coverage_mean_lag72"]) * 2.0)))))) - (data["cloud_coverage_mean_lag72"]))))))) + ((-1.0*((np.tanh((np.tanh(((((7.0)) - (data["building_median"])))))))))))/2.0)) +

            1.0*np.tanh(((((((np.where(np.where(data["year_built"] < -9998, data["year_built"], data["cloud_coverage_mean_lag3"] ) > -9998, ((((np.where(data["dew_temperature_mean_lag72"] > -9998, data["cloud_coverage_mean_lag3"], data["cloud_coverage_mean_lag72"] )) / 2.0)) / 2.0), (-1.0*((np.where(data["cloud_coverage_mean_lag72"] > -9998, ((((data["cloud_coverage_mean_lag72"]) / 2.0)) / 2.0), data["building_median"] )))) )) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh(np.tanh(((((((((((((np.tanh((((data["weekend"]) - (1.0))))) - (data["weekend"]))) + (((2.0) + (((np.tanh((data["weekend"]))) - (np.tanh((((data["dew_temperature_mean_lag3"]) + (((data["dew_temperature_mean_lag72"]) * 2.0)))))))))))/2.0)) / 2.0)) / 2.0)) / 2.0)))) +

            0.766976*np.tanh(np.tanh((np.tanh((np.where(((((((((((data["primary_use"]) / 2.0)) / 2.0)) / 2.0)) * (data["square_feet"]))) * (np.tanh((((data["primary_use"]) - ((9.45051860809326172))))))) > -9998, ((data["primary_use"]) / 2.0), 0.0 )))))) +

            0.940401*np.tanh(((np.where(data["dew_temperature_mean_lag72"] < -9998, 0.0, np.tanh((np.tanh((np.tanh((((np.where(data["cloud_coverage_mean_lag72"] < -9998, ((((data["dew_temperature_mean_lag72"]) / 2.0)) / 2.0), ((np.tanh(((-1.0*((((data["cloud_coverage_mean_lag72"]) / 2.0))))))) / 2.0) )) / 2.0))))))) )) / 2.0)))



def GPMeter2(data):

    return (5.121822 +

            1.0*np.tanh(((((((((((((data["building_median"]) - (((3.0) - (0.0))))) * 2.0)) * 2.0)) + (np.where(((np.tanh((data["air_temperature"]))) + (data["air_temperature_max_lag3"])) < -9998, (-1.0*((data["building_median"]))), (-1.0*(((((data["air_temperature_max_lag3"]) + (((np.tanh((data["air_temperature"]))) - (0.0))))/2.0)))) )))) / 2.0)) / 2.0)) +

            1.0*np.tanh(((np.where(data["cloud_coverage_mean_lag72"] < -9998, ((-2.0) + ((-1.0*((data["air_temperature_min_lag72"]))))), (((-3.0) + ((((-2.0) + (((((((((-3.0) + (np.where(-3.0 < -9998, data["wind_speed_mean_lag72"], data["cloud_coverage_mean_lag72"] )))/2.0)) + (((data["building_median"]) * 2.0)))/2.0)) * 2.0)))/2.0)))/2.0) )) - (np.tanh((np.tanh((data["air_temperature_min_lag72"]))))))) +

            0.929165*np.tanh(((np.where(data["cloud_coverage"] < -9998, data["cloud_coverage"], ((((((((data["cloud_coverage_mean_lag72"]) * (((((((data["building_median"]) * (((((((((data["building_median"]) * (((((((data["building_median"]) / 2.0)) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) * 2.0)))) / 2.0)) / 2.0)))) / 2.0)) / 2.0)) / 2.0) )) / 2.0)) +

            1.0*np.tanh((((-1.0*((((np.where(data["cloud_coverage_mean_lag72"] < -9998, data["air_temperature_min_lag72"], (-1.0*((np.tanh((((((((data["year_built"]) - ((((((((-1.0*((((data["year_built"]) / 2.0))))) * (data["air_temperature_max_lag72"]))) - (np.where(data["year_built"] > -9998, data["air_temperature_std_lag72"], data["air_temperature_max_lag3"] )))) / 2.0)))) - (data["cloud_coverage_mean_lag72"]))) / 2.0)))))) )) / 2.0))))) / 2.0)) +

            0.899365*np.tanh(((-1.0) + ((((((((data["building_median"]) + (-3.0))) + ((((np.tanh(((((-3.0) + (((((((data["wind_direction"]) + (np.where(2.0 > -9998, 3.0, data["wind_direction"] )))/2.0)) + (((data["cloud_coverage"]) + (data["air_temperature_mean_lag3"]))))/2.0)))/2.0)))) + (3.0))/2.0)))/2.0)) / 2.0)))) +

            1.0*np.tanh(np.tanh((np.where(data["sea_level_pressure_mean_lag72"] < -9998, np.where(data["dew_temperature_mean_lag3"] > -9998, data["dew_temperature_mean_lag3"], np.where(data["weekend"] > -9998, data["wind_direction"], -1.0 ) ), (((((-1.0) + (np.tanh((data["building_median"]))))/2.0)) * ((((((((data["dew_temperature_mean_lag3"]) + (np.tanh((-1.0))))/2.0)) * ((3.0)))) * ((3.0))))) )))) +

            1.0*np.tanh(((((np.tanh((np.where(data["cloud_coverage_mean_lag3"] > -9998, ((((((np.tanh((data["air_temperature_min_lag72"]))) + ((((-1.0*((data["air_temperature_min_lag72"])))) / 2.0)))/2.0)) + (((((np.tanh((data["air_temperature_min_lag72"]))) * 2.0)) * 2.0)))/2.0), (((np.tanh((((data["air_temperature_min_lag72"]) * 2.0)))) + ((((-1.0*((data["air_temperature_min_lag72"])))) / 2.0)))/2.0) )))) / 2.0)) / 2.0)) +

            0.0*np.tanh(0.0) +

            0.863214*np.tanh(np.where(((data["air_temperature_mean_lag72"]) + (data["year_built"])) < -9998, 0.0, (((((((-1.0*((((((((((((((((data["weekend"]) + ((-1.0*((((((data["air_temperature_max_lag3"]) + ((-1.0*((data["air_temperature_mean_lag72"])))))) * 2.0))))))) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0)) / 2.0))))) * 2.0)) / 2.0)) / 2.0) )) +

            0.0*np.tanh(0.0))



def GPMeter3(data):

    return (3.333021 +

            1.0*np.tanh((-1.0*((((((((data["air_temperature_mean_lag72"]) - (((np.where(data["cloud_coverage_mean_lag72"] > -9998, data["building_median"], 2.0 )) * (((data["building_median"]) + (np.tanh((((data["precip_depth_1_hr_mean_lag72"]) - (data["air_temperature"]))))))))))) / 2.0)) / 2.0))))) +

            1.0*np.tanh(np.tanh(((((np.where(data["cloud_coverage_mean_lag3"] < -9998, ((data["wind_speed_mean_lag72"]) - (data["air_temperature_min_lag3"])), ((((((((data["building_id"]) + (data["building_median"]))) + (data["building_median"]))) - (((data["sea_level_pressure"]) + (data["air_temperature_min_lag3"]))))) * 2.0) )) + (data["site_id"]))/2.0)))) +

            0.999023*np.tanh(((((((((data["building_median"]) + ((-1.0*(((5.0))))))/2.0)) - (np.where(data["building_median"] < -9998, (4.0), np.tanh((np.tanh((((((((data["dew_temperature_mean_lag3"]) / 2.0)) / 2.0)) / 2.0))))) )))) + ((((data["building_median"]) + ((-1.0*(((4.0))))))/2.0)))/2.0)) +

            1.0*np.tanh(np.tanh((((((1.0) + ((-1.0*((np.tanh(((((data["air_temperature_mean_lag3"]) + (((((data["precip_depth_1_hr_mean_lag3"]) * (np.where(data["year_built"] < -9998, data["air_temperature_std_lag72"], ((data["hour"]) - (((((((data["cloud_coverage"]) * (data["air_temperature_mean_lag3"]))) - ((((-1.0*((data["air_temperature_mean_lag3"])))) * 2.0)))) * 2.0))) )))) * 2.0)))/2.0))))))))) / 2.0)))) +

            0.899853*np.tanh(((np.tanh((((np.tanh((np.where(data["cloud_coverage"] < -9998, data["year_built"], ((np.where(data["precip_depth_1_hr_mean_lag72"] < -9998, ((((data["primary_use"]) + (np.tanh(((-1.0*((data["year_built"])))))))) / 2.0), np.tanh((data["year_built"])) )) + ((-1.0*((((data["primary_use"]) + (-1.0))))))) )))) / 2.0)))) / 2.0)) +

            1.0*np.tanh(np.where(data["cloud_coverage_mean_lag3"] > -9998, 0.0, np.where(np.where(data["year_built"] > -9998, data["cloud_coverage_mean_lag3"], data["air_temperature_mean_lag72"] ) > -9998, ((((np.tanh((np.tanh((((np.tanh((((np.tanh((data["year_built"]))) / 2.0)))) / 2.0)))))) / 2.0)) / 2.0), np.tanh((np.where(data["sea_level_pressure_mean_lag3"] > -9998, (-1.0*((np.tanh((data["air_temperature_mean_lag72"]))))), data["year_built"] ))) ) )) +

            0.986810*np.tanh(((data["primary_use"]) * (np.tanh((((((((data["primary_use"]) / 2.0)) / 2.0)) * (((((data["primary_use"]) / 2.0)) * ((((0.05702735483646393)) * (np.where(data["air_temperature_mean_lag3"] < -9998, data["wind_direction_mean_lag3"], np.tanh((np.tanh((np.where(np.where(0.0 < -9998, 0.0, data["precip_depth_1_hr"] ) < -9998, data["precip_depth_1_hr"], 0.0 ))))) )))))))))))) +

            0.831461*np.tanh(np.where((((((data["air_temperature_min_lag3"]) / 2.0)) + (((data["precip_depth_1_hr_mean_lag72"]) * (((np.tanh((((((data["air_temperature_max_lag3"]) * 2.0)) * 2.0)))) * 2.0)))))/2.0) < -9998, np.tanh((((data["air_temperature_max_lag3"]) / 2.0))), ((0.0) / 2.0) )) +

            0.893991*np.tanh(((((((np.tanh((((np.tanh(((-1.0*((data["air_temperature_max_lag3"])))))) + (((((((data["sea_level_pressure"]) * (data["primary_use"]))) + ((-1.0*(((((((((np.tanh((((data["primary_use"]) + (data["precip_depth_1_hr_mean_lag3"]))))) / 2.0)) + (data["air_temperature_max_lag3"]))/2.0)) * 2.0))))))) + ((((6.37950325012207031)) * 2.0)))))))) / 2.0)) / 2.0)) / 2.0)) +

            1.0*np.tanh((-1.0*((((np.where((-1.0*((data["square_feet"]))) < -9998, np.tanh((np.where(data["precip_depth_1_hr_mean_lag72"] > -9998, 0.0, ((((2.74675083160400391)) + ((((((((((data["dew_temperature_mean_lag3"]) + (data["square_feet"]))/2.0)) + (((((((((data["dew_temperature_mean_lag3"]) / 2.0)) / 2.0)) / 2.0)) - (3.0))))/2.0)) + (data["precip_depth_1_hr_mean_lag72"]))/2.0)))/2.0) ))), data["square_feet"] )) / 2.0))))))
target_meter = 0

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

x0 = pd.DataFrame()

x0['target'] = GPMeter0(X_train.astype('float32').fillna(-9999)).values

x0['prediction'] = y_train

x0['meter'] = target_meter

del X_train, y_train

gc.collect()

target_meter = 1

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

x1 = pd.DataFrame()

x1['target'] = GPMeter1(X_train.astype('float32').fillna(-9999)).values

x1['prediction'] = y_train

x1['meter'] = target_meter

del X_train, y_train

gc.collect()

target_meter = 2

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

x2 = pd.DataFrame()

x2['target'] = GPMeter2(X_train.astype('float32').fillna(-9999)).values

x2['prediction'] = y_train

x2['meter'] = target_meter

del X_train, y_train

gc.collect()

target_meter = 3

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

x3 = pd.DataFrame()

x3['target'] = GPMeter3(X_train.astype('float32').fillna(-9999)).values

x3['prediction'] = y_train

x3['meter'] = target_meter

del X_train, y_train

gc.collect()

x = pd.concat([x0,x1,x2,x3])

del x0, x1, x2, x3

gc.collect()
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(x.target,x.prediction))
_ = sns.distplot(x.target)
_ = sns.distplot(x.prediction)