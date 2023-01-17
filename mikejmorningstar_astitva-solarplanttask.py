# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # used to plot visual representation

import json #used for Pretty Printing dict later

import seaborn as sns

import plotly.graph_objs as go

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pg1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_wsd1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df_pg2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_wsd2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
print(df_pg1.columns)

print(df_wsd1.columns)

print()

print(df_pg1.describe())

print()

print(df_wsd1.describe())
print('Number of Inverters at Station 1: ' + str(df_pg1['SOURCE_KEY'].nunique()))

print('Number of Inverters at Station 2: ' + str(df_pg2['SOURCE_KEY'].nunique()))

print('Total Number of Inverters at Stations 1 and 2: ' + str(df_pg1['SOURCE_KEY'].nunique() + df_pg2['SOURCE_KEY'].nunique()))
print('Mean of Daily Yield at Station 1: ' + str(df_pg1['DAILY_YIELD'].mean()))

print('Mean of Daily Yield at Station 2: ' + str(df_pg2['DAILY_YIELD'].mean()))

print('Mean of Daily Yield across the Station 1 and 2: ' + str(pd.concat((df_pg1, df_pg2))['DAILY_YIELD'].mean()))
df_pg1['DATE_TIME'] = pd.to_datetime(df_pg1['DATE_TIME'])

df_wsd1['DATE_TIME'] = pd.to_datetime(df_wsd1['DATE_TIME'], format = '%Y-%m-%d %H:%M')

df_pg2['DATE_TIME'] = pd.to_datetime(df_pg2['DATE_TIME'])

df_wsd2['DATE_TIME'] = pd.to_datetime(df_wsd2['DATE_TIME'], format = '%Y-%m-%d %H:%M')

df_pg1['DATE'] = df_pg1['DATE_TIME'].dt.date

df_pg1['TIME'] = df_pg1['DATE_TIME'].dt.time

df_pg2['DATE'] = df_pg2['DATE_TIME'].dt.date

df_pg2['TIME'] = df_pg2['DATE_TIME'].dt.time

df_wsd1['DATE'] = df_wsd1['DATE_TIME'].dt.date

df_wsd1['TIME'] = df_wsd1['DATE_TIME'].dt.time

df_wsd2['DATE'] = df_wsd2['DATE_TIME'].dt.date

df_wsd2['TIME'] = df_wsd2['DATE_TIME'].dt.time
print('Total Irridation per day on Station 1 and 2:')

df_grp_irr1 = df_wsd1.groupby([df_wsd1['DATE']])['IRRADIATION'].sum()

df_grp_irr2 = df_wsd2.groupby([df_wsd2['DATE']])['IRRADIATION'].sum()

print(pd.concat([df_grp_irr1, df_grp_irr2], axis = 1))
print('Maximum Ambient Temperature at Station 1: ' + str(df_wsd1['AMBIENT_TEMPERATURE'].max()))

print('Maximum Ambient Temperature at Station 2: ' + str(df_wsd2['AMBIENT_TEMPERATURE'].max()))

print()

print('Maximum Module Temperature at Station 1:' + str(df_wsd1['MODULE_TEMPERATURE'].max()))

print('Maximum Module Temperature at Station 2:' + str(df_wsd2['MODULE_TEMPERATURE'].max()))
df_grp_ac1_max = df_pg1.groupby([df_pg1['DATE']])['AC_POWER'].max()

df_grp_dc1_max = df_pg1.groupby([df_pg1['DATE']])['DC_POWER'].max()

df_grp_ac2_max = df_pg2.groupby([df_pg2['DATE']])['AC_POWER'].max()

df_grp_dc2_max = df_pg2.groupby([df_pg2['DATE']])['DC_POWER'].max()

df_grp_ac1_min = df_pg1.groupby([df_pg1['DATE']])['AC_POWER'].min()

df_grp_dc1_min = df_pg1.groupby([df_pg1['DATE']])['DC_POWER'].min()

df_grp_ac2_min = df_pg2.groupby([df_pg2['DATE']])['AC_POWER'].min()

df_grp_dc2_min = df_pg2.groupby([df_pg2['DATE']])['DC_POWER'].min()

df_grp_ac1_min_nz = df_pg1[df_pg1['AC_POWER'] != 0].groupby([df_pg1['DATE_TIME'].dt.date])['AC_POWER'].min()

df_grp_dc1_min_nz = df_pg1[df_pg1['DC_POWER'] != 0].groupby([df_pg1['DATE_TIME'].dt.date])['DC_POWER'].min()

df_grp_ac2_min_nz = df_pg2[df_pg2['AC_POWER'] != 0].groupby([df_pg2['DATE_TIME'].dt.date])['AC_POWER'].min()

df_grp_dc2_min_nz = df_pg2[df_pg2['DC_POWER'] != 0].groupby([df_pg2['DATE_TIME'].dt.date])['DC_POWER'].min()



print('Maximum AC and DC Power at Station 1 each day:')

print(pd.concat([df_grp_ac1_max, df_grp_dc1_max], axis = 1))

print('\nMaximum AC and DC Power at Station 2 each day:')

print(pd.concat([df_grp_ac2_max, df_grp_dc2_max], axis = 1))

print('\nMinimum AC and DC Power at Station 1 each day:')

print(pd.concat([df_grp_ac1_min, df_grp_dc1_min], axis = 1))

print('\nMinimum AC and DC Power at Station 2 each day:')

print(pd.concat([df_grp_ac2_min, df_grp_dc2_min], axis = 1))

print('\nMinimum AC and DC Power at Station 1 each day(Non Zero):')

print(pd.concat([df_grp_ac1_min_nz, df_grp_dc1_min_nz], axis = 1))

print('\nMinimum AC and DC Power at Station 2 each day(Non Zero):')

print(pd.concat([df_grp_ac2_min_nz, df_grp_dc2_min_nz], axis = 1))
print('ID of Inverter producing Maximum AC Power at Station 1: ' + (df_pg1.iloc[df_pg1['AC_POWER'].idxmax()])['SOURCE_KEY'])

print('ID of Inverter producing Maximum DC Power at Station 1: ' + (df_pg1.iloc[df_pg1['DC_POWER'].idxmax()])['SOURCE_KEY'])

print('ID of Inverter producing Maximum AC Power at Station 2: ' + (df_pg2.iloc[df_pg2['AC_POWER'].idxmax()])['SOURCE_KEY'])

print('ID of Inverter producing Maximum DC Power at Station 2: ' + (df_pg2.iloc[df_pg2['DC_POWER'].idxmax()])['SOURCE_KEY'])
dc_mean = {}

ac_mean = {}

for i in df_pg1['SOURCE_KEY'].unique():

    dc_mean[i] = df_pg1[df_pg1['SOURCE_KEY'] == i]['DC_POWER'].mean()

    ac_mean[i] = df_pg1[df_pg1['SOURCE_KEY'] == i]['AC_POWER'].mean()

dc_mean1 = {sk: m for sk, m in sorted(dc_mean.items(), key = lambda item: item[1], reverse = True)}

ac_mean1 = {sk: m for sk, m in sorted(ac_mean.items(), key = lambda item: item[1], reverse = True)}

for i in df_pg2['SOURCE_KEY'].unique():

    dc_mean[i] = df_pg2[df_pg2['SOURCE_KEY'] == i]['DC_POWER'].mean()

    ac_mean[i] = df_pg2[df_pg2['SOURCE_KEY'] == i]['AC_POWER'].mean()

dc_mean2 = {sk: m for sk, m in sorted(dc_mean.items(), key = lambda item: item[1], reverse = True)}

ac_mean2 = {sk: m for sk, m in sorted(ac_mean.items(), key = lambda item: item[1], reverse = True)}



print('Inverter Rank based on Mean DC Power at Station 1:\n')

print(json.dumps(dc_mean1, indent = 4))

print('\nInverter Rank based on Mean AC Power at Station 1:\n')

print(json.dumps(ac_mean1, indent = 4))

print('\nInverter Rank based on Mean DC Power at Station 2:\n')

print(json.dumps(dc_mean2, indent = 4))

print('\nInverter Rank based on Mean AC Power at Station 2:\n')

print(json.dumps(ac_mean2, indent = 4))
print('Number of entries: ' + str(len(df_pg1.index)) + '\nExpected entries: ' + str(34*22*24*4))