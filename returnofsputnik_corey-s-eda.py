import numpy as np

import pandas as pd

import os, gc

from tqdm import tqdm_notebook
tr = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

te = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
tr.head()
te.head()
tr.groupby(['building_id','meter']).size()
tr['meter'].value_counts()
# Zoom in onto one building, and one meter.

tr.query('building_id==0 & meter==0')
meta = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')
meta
for col in meta.columns:

    if meta[col].isnull().sum() > 0: # If you have any rows with NaN in it...

        print(col)
primary_use_to_id = pd.concat([pd.Series(meta['primary_use'].unique()), (pd.Series(meta['primary_use'].unique())).astype('category').cat.codes], axis=1).set_index(0).to_dict()[1]

id_to_primary_use = pd.concat([pd.Series(meta['primary_use'].unique()), (pd.Series(meta['primary_use'].unique())).astype('category').cat.codes], axis=1).set_index(1).to_dict()[0]
primary_use_to_id
meta['primary_use'] = meta['primary_use'].map(primary_use_to_id).astype('int32')
te['meter_reading'] = -1

tr['row_id'] = -1



tr_te = pd.concat([tr,te],axis=0,sort=True)

del tr,te; gc.collect()
timestamp_to_id = pd.concat([pd.Series(tr_te['timestamp'].unique()), (pd.Series(tr_te['timestamp'].unique())).astype('category').cat.codes], axis=1).set_index(0).to_dict()[1]

id_to_timestamp = pd.concat([pd.Series(tr_te['timestamp'].unique()), (pd.Series(tr_te['timestamp'].unique())).astype('category').cat.codes], axis=1).set_index(1).to_dict()[0]
# The number of unique timestamps we have in train + test.

print(len(timestamp_to_id))
tr_te.head()
tr_te['timestamp'] = tr_te['timestamp'].map(timestamp_to_id).astype('int64')
tr_te.head()
gc.collect()
print('Original shape is',tr_te.shape) 

tr_te = tr_te.merge(meta, on='building_id')

print('Shape after merging is',tr_te.shape) 
tr_te.head()
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=20,15

import seaborn as sns
sns.kdeplot(meta['square_feet'], label='square_feet', shade=True, kernel='epa')
sns.kdeplot(meta['year_built'], label='year_built', shade=True, kernel='epa')
meta['year_built'].max(), meta['year_built'].min()
sns.kdeplot(meta['floor_count'], label='floor_count', shade=True, kernel='epa')
# sns.scatterplot(tr_te['square_feet'], tr_te['meter_reading'])

# sns.scatterplot(tr_te['year_built'], tr_te['meter_reading'])

# sns.scatterplot(tr_te['floor_count'], tr_te['meter_reading'])
weather_tr = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_tr['timestamp'] = weather_tr['timestamp'].map(timestamp_to_id).astype('int64')

weather_te = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

weather_te['timestamp'] = weather_te['timestamp'].map(timestamp_to_id).astype('int64')



weather = pd.concat([weather_tr,weather_te],axis=0)

del weather_tr, weather_te; gc.collect()
tr_te['building_id'] = tr_te['building_id'].astype('int8')

tr_te['meter'] = tr_te['meter'].astype('int8')

tr_te['row_id'] = tr_te['row_id'].astype('int32')

tr_te['timestamp'] = tr_te['timestamp'].astype('int32')

tr_te['site_id'] = tr_te['site_id'].astype('int8')

tr_te['primary_use'] = tr_te['primary_use'].astype('int8')

tr_te['square_feet'] = tr_te['square_feet'].astype('int32')

tr_te['year_built'] = tr_te['year_built'].astype('float16')

tr_te['floor_count'] = tr_te['floor_count'].astype('float16')



weather['site_id'] = weather['site_id'].astype('int8')

weather['timestamp'] = weather['timestamp'].astype('int32')

weather['air_temperature'] = weather['air_temperature'].astype('float16')

weather['cloud_coverage'] = weather['cloud_coverage'].astype('float16')

weather['dew_temperature'] = weather['dew_temperature'].astype('float16')

weather['precip_depth_1_hr'] = weather['precip_depth_1_hr'].astype('float16')

weather['sea_level_pressure'] = weather['sea_level_pressure'].astype('float16')

weather['wind_direction'] = weather['wind_direction'].astype('float16')

weather['wind_speed'] = weather['wind_speed'].astype('float16')



gc.collect()
print('Original shape is',tr_te.shape) 

tr_te = tr_te.merge(weather, on=['site_id','timestamp'], how='left')

print('Shape after merging is',tr_te.shape) 
tr_te.dtypes
primary_use_to_id
tr_te.head()