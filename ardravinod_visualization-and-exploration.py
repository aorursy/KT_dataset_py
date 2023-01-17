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
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df_pwea1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df_pwea2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
import matplotlib.pyplot as plt
df_pwea1.info()
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')

df_pwea1['DATE_TIME'] = pd.to_datetime(df_pwea1['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_pwea1['DATE'] = df_pwea1['DATE_TIME'].apply(lambda x:x.date())

df_pwea1['TIME'] = df_pwea1['DATE_TIME'].apply(lambda x:x.time())
df_pwea1['DATE'] = pd.to_datetime(df_pwea1['DATE'],format = '%Y-%m-%d')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour

df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_pwea1.info()
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())

df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1.info()
df_pwea1.nunique()
plt.plot(df_pwea1['IRRADIATION'],df_pwea1['DATE_TIME'],label = 'irradiation per day')

plt.xlabel('irradiation')

plt.ylabel('date time')

plt.legend()

plt.show()
plt.plot(df_pgen1['DC_POWER'],df_pgen1['DATE_TIME'])

plt.show()
plt.figure(figsize=(12,8))

plt.plot(df_pgen1['DATE_TIME'],df_pgen1['AC_POWER'])

plt.show()
plt.plot(df_pgen1['DATE_TIME'],df_pgen1['TOTAL_YIELD'])

plt.show()
plt.figure(figsize=(12,8))

plt.plot(df_pgen1['TOTAL_YIELD'],df_pgen1['DC_POWER'].rolling(window=20).mean())

plt.show()
# normal plot

plt.figure(figsize=(12,8))

plt.plot(df_pwea1['DATE_TIME'],df_pwea1['AMBIENT_TEMPERATURE'],label ='AMBIENT',c='cyan')

plt.plot(df_pwea1['DATE_TIME'],df_pwea1['MODULE_TEMPERATURE'],label ='MODULE',c='orange')

plt.plot(df_pwea1['DATE_TIME'],df_pwea1['MODULE_TEMPERATURE']-df_pwea1['AMBIENT_TEMPERATURE'],label ='DIFFERENCE',c='k')

plt.grid()

plt.margins(0.05)

plt.legend()
# using rolling(window=) to make the graph more readable

plt.figure(figsize=(20,10))

plt.plot(df_pwea1['DATE_TIME'],df_pwea1['AMBIENT_TEMPERATURE'].rolling(window=20).mean(),label ='AMBIENT',c='cyan')

plt.plot(df_pwea1['DATE_TIME'],df_pwea1['MODULE_TEMPERATURE'].rolling(window=20).mean(),label ='MODULE',c='orange')

plt.plot(df_pwea1['DATE_TIME'],(df_pwea1['MODULE_TEMPERATURE']-df_pwea1['AMBIENT_TEMPERATURE']).rolling(window=20).mean(),label ='DIFFERENCE',c='k')

plt.grid()

plt.margins(0.05)

plt.legend()
# scatter plot without using plt.scatter

plt.plot(df_pwea1['AMBIENT_TEMPERATURE'],df_pwea1['MODULE_TEMPERATURE'], marker = 'o',linestyle='')

plt.xlabel('AMBIENT TEMP')

plt.ylabel('MODULE TEMP')

plt.show()
# using plt.scatter

plt.scatter(df_pwea1['AMBIENT_TEMPERATURE'],df_pwea1['MODULE_TEMPERATURE'] , c='r',alpha = 0.5)

plt.xlabel('AMBIENT TEMP')

plt.ylabel('MODULE TEMP')

plt.show()
plt.figure(figsize=(20,10))

plt.scatter(df_pwea1['AMBIENT_TEMPERATURE'],df_pwea1['MODULE_TEMPERATURE'],c=df_pwea1['AMBIENT_TEMPERATURE'],alpha =0.5)

plt.show()
df_pgen1.info()
plt.plot(df_pgen1['DATE_TIME'],df_pgen1['SOURCE_KEY'],linestyle='',marker = 'o', c='orange',alpha = 0.5)

plt.grid()

plt.xlabel('date')

plt.ylabel('source key')

plt.show()
plt.plot(df_pgen1['HOUR'],df_pgen1['AC_POWER'],linestyle='',c='k',marker='o',alpha=0.5)

plt.xlabel('hours')

plt.ylabel('ac power')

plt.show()
dates = df_pwea1['DATE'].unique()

dates
plt.figure(figsize=(20,10))

for date in dates:

    data = df_pwea1[df_pwea1['DATE'] == date][df_pwea1['IRRADIATION']>0]

    plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'],marker = 'o',linestyle='',label = pd.to_datetime(date,format = '%Y-%m-%d').date())

plt.legend()
plt.figure(figsize=(12,8))

plt.scatter(df_pwea1['IRRADIATION'],df_pwea1['MODULE_TEMPERATURE']-df_pwea1['AMBIENT_TEMPERATURE'],c='k')

plt.xlabel('irradiation')

plt.ylabel('diff between temp')

plt.show()
plt.figure(figsize=(12,8))

plt.scatter(df_pwea1['IRRADIATION'],df_pwea1['MODULE_TEMPERATURE']+df_pwea1['AMBIENT_TEMPERATURE'], c='orange')

plt.xlabel('irradiation')

plt.ylabel('sum of temp')

plt.show()