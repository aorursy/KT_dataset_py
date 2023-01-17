import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

from datetime import datetime

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')
weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])

weather_test['date'] = weather_test['timestamp'].apply(datetime.date)

weather_test['time'] = weather_test['timestamp'].apply(datetime.time)
for j in ['air_temperature']:

    temp = weather_test.groupby(['site_id','date','time'])[j].agg(pd.Series.mode)

    temp_list = []

    for i in weather_test.index:

        if np.isnan(weather_test.loc[i][j]):

            try:

                temp1 = temp[[weather_test.loc[i].site_id,weather_test.loc[i].date,weather_test.loc[i].time]]

                temp_list.append(temp1)

            except:

                temp_list.append(np.nan)

        else:

            temp_list.append(weather_test.loc[i][j])

    weather_test[j] = temp_list

    temp1 = []

    for i in weather_test.index:

        if type(weather_test.loc[i].air_temperature) == np.float64:

            temp = weather_test.loc[i].air_temperature

            temp1.append(temp)

        else:

            temp2 = 0

            temp3 = 0

            for i in weather_test.loc[i].air_temperature:

                if type(i) == np.float64:

                    temp2 += i

                    temp3 += 1

            temp1.append(temp2/temp3)

    weather_test[j] = temp1

    del temp1

    del temp

    del temp_list
weather_test['air_temperature'].isna().sum()
gc.collect()
for j in ['dew_temperature']:

    temp = weather_test.groupby(['site_id','date','time'])[j].agg(pd.Series.mode)

    temp_list = []

    for i in weather_test.index:

        if np.isnan(weather_test.loc[i][j]):

            try:

                temp1 = temp[[weather_test.loc[i].site_id,weather_test.loc[i].date,weather_test.loc[i].time]]

                temp_list.append(temp1)

            except:

                temp_list.append(np.nan)

        else:

            temp_list.append(weather_test.loc[i][j])

    weather_test[j] = temp_list

    temp1 = []

    for i in weather_test.index:

        if type(weather_test.loc[i].air_temperature) == np.float64:

            temp = weather_test.loc[i].air_temperature

            temp1.append(temp)

        else:

            temp2 = 0

            temp3 = 0

            for i in weather_test.loc[i].air_temperature:

                if type(i) == np.float64:

                    temp2 += i

                    temp3 += 1

            temp1.append(temp2/temp3)

    weather_test[j] = temp1

    del temp1

    del temp

    del temp_list
weather_test['dew_temperature'].isna().sum()
for j in ['wind_speed']:

    temp = weather_test.groupby(['site_id','date','time'])[j].agg(pd.Series.mode)

    temp_list = []

    for i in weather_test.index:

        if np.isnan(weather_test.loc[i][j]):

            try:

                temp1 = temp[[weather_test.loc[i].site_id,weather_test.loc[i].date,weather_test.loc[i].time]]

                temp_list.append(temp1)

            except:

                temp_list.append(np.nan)

        else:

            temp_list.append(weather_test.loc[i][j])

    weather_test[j] = temp_list

    temp1 = []

    for i in weather_test.index:

        if type(weather_test.loc[i].air_temperature) == np.float64:

            temp = weather_test.loc[i].air_temperature

            temp1.append(temp)

        else:

            temp2 = 0

            temp3 = 0

            for i in weather_test.loc[i].air_temperature:

                if type(i) == np.float64:

                    temp2 += i

                    temp3 += 1

            temp1.append(temp2/temp3)

    weather_test[j] = temp1

    del temp1

    del temp

    del temp_list
weather_test['wind_speed'].isna().sum()
for j in ['wind_direction']:

    temp = weather_test.groupby(['site_id','date','time'])[j].agg(pd.Series.mode)

    temp_list = []

    for i in weather_test.index:

        if np.isnan(weather_test.loc[i][j]):

            try:

                temp1 = temp[[weather_test.loc[i].site_id,weather_test.loc[i].date,weather_test.loc[i].time]]

                temp_list.append(temp1)

            except:

                temp_list.append(np.nan)

        else:

            temp_list.append(weather_test.loc[i][j])

    weather_test[j] = temp_list

    temp1 = []

    for i in weather_test.index:

        if type(weather_test.loc[i].air_temperature) == np.float64:

            temp = weather_test.loc[i].air_temperature

            temp1.append(temp)

        else:

            temp2 = 0

            temp3 = 0

            for i in weather_test.loc[i].air_temperature:

                if type(i) == np.float64:

                    temp2 += i

                    temp3 += 1

            temp1.append(temp2/temp3)

    weather_test[j] = temp1

    del temp1

    del temp

    del temp_list
weather_test['wind_direction'].isna().sum()
for j in ['sea_level_pressure']:

    temp = weather_test.groupby(['site_id','date','time'])[j].agg(pd.Series.mode)

    temp_list = []

    for i in weather_test.index:

        if np.isnan(weather_test.loc[i][j]):

            try:

                temp1 = temp[[weather_test.loc[i].site_id,weather_test.loc[i].date,weather_test.loc[i].time]]

                temp_list.append(temp1)

            except:

                temp_list.append(np.nan)

        else:

            temp_list.append(weather_test.loc[i][j])

    weather_test[j] = temp_list

    temp1 = []

    for i in weather_test.index:

        if type(weather_test.loc[i].air_temperature) == np.float64:

            temp = weather_test.loc[i].air_temperature

            temp1.append(temp)

        else:

            temp2 = 0

            temp3 = 0

            for i in weather_test.loc[i].air_temperature:

                if type(i) == np.float64:

                    temp2 += i

                    temp3 += 1

            temp1.append(temp2/temp3)

    weather_test[j] = temp1

    del temp1

    del temp

    del temp_list
weather_test['sea_level_pressure'].isna().sum()
del weather_test['cloud_coverage']
weather_test.to_csv('cleaned_weather_test.csv',index=False)