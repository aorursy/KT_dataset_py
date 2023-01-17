# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import datetime
import gc
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

path='/kaggle/input/ashrae-energy-prediction/'
#weather train
weather_train=pd.read_csv(path+'weather_train.csv')
display(weather_train.shape)
display(weather_train.head())
display(weather_train.columns)
display(weather_train.dtypes)
#check index
display(weather_train['site_id'].is_unique)
#wearher test
weather_test=pd.read_csv(path+'weather_test.csv')
display(weather_test.shape)
display(weather_test.head())
display(weather_test.columns)
display(weather_test.dtypes)
# check for index
display(weather_test['site_id'].is_unique)
#Merge weather_train & Weather_test
weather_merged=weather_train.append(weather_test)
weather_merged['timestamp']=pd.to_datetime(weather_merged['timestamp'])
weather_merged.set_index('timestamp',inplace=True)
display(weather_merged.shape)
display(weather_merged.columns)
display(weather_merged.head())
#extract wind speed from weather data
wind_speed_pivot=weather_merged.reset_index().pivot_table(index='timestamp',columns='site_id',values='wind_speed')
wind_speed_pivot.columns='site_'+wind_speed_pivot.columns.astype('str')
wind_speed_pivot
#load external wind speed data
speed_external = pd.read_csv("../input/historical-hourly-weather-data/wind_speed.csv")
speed_external['datetime'] = pd.to_datetime(speed_external['datetime'])
speed_external.set_index('datetime', inplace=True)

speed_external = speed_external.merge(wind_speed_pivot, left_index=True, right_index=True, how='inner')
speed_external = speed_external.dropna()

speed_external
#calculate correlations between sites
df_corr = speed_external.corr(method='spearman')
list_site = wind_speed_pivot.columns
df_corr = df_corr[list_site]
df_corr = df_corr.drop(list_site)
df_corr
#sns heat map
fig, ax = plt.subplots(figsize=(30,15))   
sns.heatmap(df_corr, annot=True, cmap="YlGnBu", vmin=0.08, vmax=0.10)
#Get cities!
df_findCity = pd.concat([df_corr.idxmax(),df_corr.max()], axis=1).reset_index().rename(columns={'index':'site',0:'city',1:'corr'})
df_findCity
#compare sites & cities in plot
for city, site, corr in zip(df_findCity['city'],df_findCity['site'],df_findCity['corr']):
    if corr > 0.08:
        print('City: ' + city)
        print('Site: ' + site)   
        speed_external[[city,site]].loc['2016'].plot(figsize=(15,8), alpha=0.5)
        plt.show()
#extract wind speed from weather data
wind_direction_pivot=weather_merged.reset_index().pivot_table(index='timestamp',columns='site_id',values='wind_direction')
wind_direction_pivot.columns='site_'+wind_direction_pivot.columns.astype('str')
wind_direction_pivot
#load external wind direction data
direction_external = pd.read_csv("../input/historical-hourly-weather-data/wind_direction.csv")
direction_external['datetime'] = pd.to_datetime(direction_external['datetime'])
direction_external.set_index('datetime', inplace=True)

direction_external = direction_external.merge(wind_direction_pivot, left_index=True, right_index=True, how='inner')
direction_external = direction_external.dropna()

direction_external
#calculate correlations between sites
df_corr = direction_external.corr(method='spearman')
list_site = wind_direction_pivot.columns
df_corr = df_corr[list_site]
df_corr = df_corr.drop(list_site)
df_corr
#sns heat map
fig, ax = plt.subplots(figsize=(30,15))   
sns.heatmap(df_corr, annot=True, cmap="YlGnBu", vmin=0,vmax=0.3)
#Get cities!
df_findCity = pd.concat([df_corr.idxmax(),df_corr.max()], axis=1).reset_index().rename(columns={'index':'site',0:'city',1:'corr'})
df_findCity
#compare sites & cities in plot
for city, site, corr in zip(df_findCity['city'],df_findCity['site'],df_findCity['corr']):
    if corr > 0.08:
        print('City: ' + city)
        print('Site: ' + site)   
        direction_external[[city,site]].loc['2016'].plot(figsize=(15,8), alpha=0.5)
        plt.show()
#extract wind speed from weather data
sea_level_pivot=weather_merged.reset_index().pivot_table(index='timestamp',columns='site_id',values='sea_level_pressure')
sea_level_pivot.columns='site_' +sea_level_pivot.columns.astype('str')
sea_level_pivot
#load external wind direction data
pressure_external = pd.read_csv("../input/historical-hourly-weather-data/pressure.csv")
pressure_external['datetime'] = pd.to_datetime(pressure_external['datetime'])
pressure_external.set_index('datetime', inplace=True)
pressure_external=pressure_external-760
preesure_external = pressure_external.merge(sea_level_pivot, left_index=True, right_index=True, how='inner')
pressure_external = pressure_external.dropna()

pressure_external
#calculate correlations between sites
df_corr = direction_external.corr(method='spearman')
list_site = wind_direction_pivot.columns
df_corr = df_corr[list_site]
df_corr = df_corr.drop(list_site)
df_corr
#sns heat map
fig, ax = plt.subplots(figsize=(30,15))   
sns.heatmap(df_corr, annot=True, cmap="YlGnBu", vmin=0,vmax=0.6)
#Get cities!
df_findCity = pd.concat([df_corr.idxmax(),df_corr.max()], axis=1).reset_index().rename(columns={'index':'site',0:'city',1:'corr'})
df_findCity
