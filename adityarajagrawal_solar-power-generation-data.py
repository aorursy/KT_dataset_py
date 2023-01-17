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
wsd1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

gd1 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

wsd2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

gd2 = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')
wsd1
gd1
wsd2.info()
gd2
print('Mean daily yeild value of plant 1 is',gd1['DAILY_YIELD'].mean())

print('Mean daily yeild value of plant 2 is',gd2['DAILY_YIELD'].mean())

total_mean = (gd1['DAILY_YIELD'].sum() + gd2['DAILY_YIELD'].sum())/(len(gd1['DAILY_YIELD'])+len(gd2['DAILY_YIELD']))

print('Total Mean daily yeild value of plant 1 and plant 2 is',total_mean)
wsd1['DATE_TIME'] = pd.to_datetime(wsd1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

wsd2['DATE_TIME'] = pd.to_datetime(wsd2['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

gd1['DATE_TIME'] = pd.to_datetime(gd1['DATE_TIME'],format='%d-%m-%Y %H:%M')

gd2['DATE_TIME'] = pd.to_datetime(gd2['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')
wsd1['IRRADIATION'].groupby([wsd1['DATE_TIME'].dt.date]).sum()
wsd2['IRRADIATION'].groupby([wsd2['DATE_TIME'].dt.date]).sum()
print('Maximum Ambient Temperature of plant 1 is',wsd1['AMBIENT_TEMPERATURE'].max())

print('Maximum Module Temperature of plant 1 is',wsd1['MODULE_TEMPERATURE'].max())

print('Maximum Ambient Temperature of plant 2 is',wsd2['AMBIENT_TEMPERATURE'].max())

print('Maximum Module Temperature of plant 2 is',wsd2['MODULE_TEMPERATURE'].max())
print('Number of inverters in Plant 1 are',len(gd1['SOURCE_KEY'].unique()))

print('Number of inverters in Plant 2 are',len(gd2['SOURCE_KEY'].unique()))
gd1['DC_POWER'].groupby(gd1['DATE_TIME'].dt.date).max()
gd1['DC_POWER'].groupby(gd1['DATE_TIME'].dt.date).min()
gd1['AC_POWER'].groupby(gd1['DATE_TIME'].dt.date).max()
gd1['AC_POWER'].groupby(gd1['DATE_TIME'].dt.date).min()
gd2['DC_POWER'].groupby(gd2['DATE_TIME'].dt.date).max()
gd2['DC_POWER'].groupby(gd2['DATE_TIME'].dt.date).min()
gd2['AC_POWER'].groupby(gd2['DATE_TIME'].dt.date).max()
gd2['AC_POWER'].groupby(gd2['DATE_TIME'].dt.date).min()
a = gd1.set_index('SOURCE_KEY')[['DC_POWER','AC_POWER']]

b = gd2.set_index('SOURCE_KEY')[['DC_POWER','AC_POWER']]
print('Maximum DC Power is produced by inverter in Plant 1',a['DC_POWER'].idxmax())

print('Maximum AC Power is produced by inverter in Plant 1',a['AC_POWER'].idxmax())
print('Maximum DC Power is produced by inverter in Plant 2',b['DC_POWER'].idxmax())

print('Maximum AC Power is produced by inverter in Plant 2',b['AC_POWER'].idxmax())
a = (a.groupby('SOURCE_KEY').max())

b = (b.groupby('SOURCE_KEY').max())
a.reset_index(inplace=True)

a.sort_values('DC_POWER',ascending=False)['SOURCE_KEY']
b.reset_index(inplace=True)

b.sort_values('DC_POWER',ascending=False)['SOURCE_KEY']