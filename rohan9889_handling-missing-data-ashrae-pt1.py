import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from datetime import datetime

import gc

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
building_metadata = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')
building_metadata.shape
building_metadata.head()
building_metadata.shape
building_metadata['primary_use'].isna().sum()
building_metadata['square_feet'].isna().sum()
building_metadata['year_built'].isna().sum()
building_metadata['floor_count'].isna().sum()
del building_metadata['floor_count']
building_metadata.to_csv('cleaned_building_metadata.csv',index=False)
del building_metadata

gc.collect()
weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
weather_train.shape
weather_train.head()
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
weather_train['date'] = weather_train['timestamp'].apply(datetime.date)
weather_train['time'] = weather_train['timestamp'].apply(datetime.time)
weather_train['air_temperature'].isna().sum()
temp = weather_train.groupby(['site_id','date','time'])['air_temperature'].agg(pd.Series.mode)
temp
temp_list = []

for i in weather_train.index:

    if np.isnan(weather_train.loc[i].air_temperature):

        try:

            temp1 = temp[[weather_train.loc[i].site_id,weather_train.loc[i].date,weather_train.loc[i].time]]

            temp_list.append(temp1)

        except:

            temp_list.append(np.nan)

    else:

        temp_list.append(weather_train.loc[i].air_temperature)
weather_train['air_temperature'] = temp_list
temp1 = []

for i in weather_train.index:

    if type(weather_train.loc[i].air_temperature) == np.float64:

        temp = weather_train.loc[i].air_temperature

        temp1.append(temp)

    else:

        temp2 = 0

        temp3 = 0

        for i in weather_train.loc[i].air_temperature:

            if type(i) == np.float64:

                temp2 += i

                temp3 += 1

        temp1.append(temp2/temp3)
weather_train['air_temperature'] = temp1
weather_train['air_temperature'].isna().sum()
weather_train['dew_temperature'].isna().sum()
weather_train['cloud_coverage'].isna().sum()
weather_train['sea_level_pressure'].isna().sum()
weather_train['wind_direction'].isna().sum()
weather_train['wind_speed'].isna().sum()
for j in ['dew_temperature','wind_speed','wind_direction','sea_level_pressure','cloud_coverage']:

    temp = weather_train.groupby(['site_id','date','time'])[j].agg(pd.Series.mode)

    temp_list = []

    for i in weather_train.index:

        if np.isnan(weather_train.loc[i][j]):

            try:

                temp1 = temp[[weather_train.loc[i].site_id,weather_train.loc[i].date,weather_train.loc[i].time]]

                temp_list.append(temp1)

            except:

                temp_list.append(np.nan)

        else:

            temp_list.append(weather_train.loc[i][j])

    weather_train[j] = temp_list

    temp1 = []

    for i in weather_train.index:

        if type(weather_train.loc[i].air_temperature) == np.float64:

            temp = weather_train.loc[i].air_temperature

            temp1.append(temp)

        else:

            temp2 = 0

            temp3 = 0

            for i in weather_train.loc[i].air_temperature:

                if type(i) == np.float64:

                    temp2 += i

                    temp3 += 1

            temp1.append(temp2/temp3)

    weather_train[j] = temp1
weather_train['precip_depth_1_hr'].isna().sum()
del weather_train['precip_depth_1_hr']
weather_train.to_csv('cleaned_weather_train.csv',index=False)
del weather_train

gc.collect()