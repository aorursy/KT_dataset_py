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
# loading the datasets

df1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

df2 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

df3 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")

df4 = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
# cleaning the data

df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'], format ='%d-%m-%Y %H:%M')

df1['DATE'] = df1['DATE_TIME'].dt.date

df1['DATE'] = pd.to_datetime(df1['DATE'],format = '%Y-%m-%d')

df1['TIME'] = df1['DATE_TIME'].dt.time

df1['HOUR'] = pd.to_datetime(df1['TIME'],format='%H:%M:%S').dt.hour

df1['MINUTES'] = pd.to_datetime(df1['TIME'],format='%H:%M:%S').dt.minute

df1.info()



df2['DATE_TIME'] = pd.to_datetime(df2['DATE_TIME'], format ='%Y-%m-%d %H:%M')

df2['DATE'] = df2['DATE_TIME'].dt.date

df2['DATE'] = pd.to_datetime(df2['DATE'],format = '%Y-%m-%d')

df2['TIME'] = df2['DATE_TIME'].dt.time

df2['HOUR'] = pd.to_datetime(df2['TIME'],format='%H:%M:%S').dt.hour

df2['MINUTES'] = pd.to_datetime(df2['TIME'],format='%H:%M:%S').dt.minute

df2.info()



df3['DATE_TIME'] = pd.to_datetime(df3['DATE_TIME'], format ='%Y-%m-%d %H:%M')

df3['DATE'] = df3['DATE_TIME'].dt.date

df3['DATE'] = pd.to_datetime(df3['DATE'],format = '%Y-%m-%d')

df3['TIME'] = df3['DATE_TIME'].dt.time

df3['HOUR'] = pd.to_datetime(df3['TIME'],format='%H:%M:%S').dt.hour

df3['MINUTES'] = pd.to_datetime(df3['TIME'],format='%H:%M:%S').dt.minute

df3.info()



df4['DATE_TIME'] = pd.to_datetime(df4['DATE_TIME'], format ='%Y-%m-%d %H:%M')

df4['DATE'] = df4['DATE_TIME'].dt.date

df4['DATE'] = pd.to_datetime(df4['DATE'],format = '%Y-%m-%d')

df4['TIME'] = df4['DATE_TIME'].dt.time

df4['HOUR'] = pd.to_datetime(df4['TIME'],format='%H:%M:%S').dt.hour

df4['MINUTES'] = pd.to_datetime(df4['TIME'],format='%H:%M:%S').dt.minute

df4.info()
plt.subplots(1,1, figsize=(20,10))

plt.title("DC Power of both plants across different dates")

plt.plot(df1.DATE_TIME, df1.DC_POWER.rolling(window=20).mean(), label='Plant 1 DC Power', c='b')

plt.plot(df3.DATE_TIME, df3.DC_POWER.rolling(window=20).mean(), label='Plant 2 DC Power', c='r')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('DC POWER')

plt.tight_layout()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20,10))

plt.title("AC Power of both plants across different dates")

plt.plot(df1.DATE_TIME, df1.AC_POWER.rolling(window=20).mean(), label='Plant 1 AC Power', c='b')

plt.plot(df3.DATE_TIME, df3.AC_POWER.rolling(window=20).mean(), label='Plant 2 AC Power', c='r')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('AC POWER')

plt.tight_layout()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20, 10))

plt.title("PLANT 1 DC POWER VS AC POWER")

plt.plot(df1.DATE_TIME, df1.DC_POWER, marker='o', linestyle='', label='DC_POWER', c='b')

plt.plot(df1.DATE_TIME, df1.AC_POWER, marker = 'o', linestyle='', label='AC_POWER', c='r')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('AC/DC POWER')

plt.grid()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20, 10))

plt.title("PLANT 2 DC POWER VS AC POWER")

plt.plot(df3.DATE_TIME, df3.DC_POWER, marker='o', linestyle='', label='DC_POWER', c='b')

plt.plot(df3.DATE_TIME, df3.AC_POWER, marker = 'o', linestyle='', label='AC_POWER', c='r')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('AC/DC POWER')

plt.grid()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20,10))

plt.title("Ambient temperature of both plants across different dates")

plt.plot(df2.DATE_TIME, df2.AMBIENT_TEMPERATURE.rolling(window=20).mean(), label='Plant 1 Ambient temp', c='k')

plt.plot(df4.DATE_TIME, df4.AMBIENT_TEMPERATURE.rolling(window=20).mean(), label='Plant 2 Ambient temp', c='y')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('AMBIENT TEMPERATURE')

plt.grid()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20,10))

plt.title("Module temperature of both plants across different dates")

plt.plot(df2.DATE_TIME, df2.MODULE_TEMPERATURE.rolling(window=20).mean(), label='Plant 1 Module temp', c='k')

plt.plot(df4.DATE_TIME, df4.MODULE_TEMPERATURE.rolling(window=20).mean(), label='Plant 2 Module temp', c='y')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('MODULE TEMPERATURE')

plt.grid()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20, 10))

plt.title("PLANT 1 AMBIENT TEMP VS MODULE TEMP")

plt.plot(df2['DATE_TIME'], df2['AMBIENT_TEMPERATURE'], label ='Ambient temp', c ='k')

plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE'], label ='Module temp', c='y' )

plt.plot(df2['DATE_TIME'],df2['MODULE_TEMPERATURE']-df2['AMBIENT_TEMPERATURE'], label ='Difference' , c ='r')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('AMBIENT/MODULE TEMPERATURE')

plt.grid()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20, 10))

plt.title("PLANT 2 AMBIENT TEMP VS MODULE TEMP")

plt.plot(df4['DATE_TIME'], df4['AMBIENT_TEMPERATURE'], label ='Ambient temp', c ='k')

plt.plot(df4['DATE_TIME'],df4['MODULE_TEMPERATURE'], label ='Module temp', c='y' )

plt.plot(df4['DATE_TIME'],df4['MODULE_TEMPERATURE']-df4['AMBIENT_TEMPERATURE'], label ='Difference' , c ='r')

plt.xticks(rotation = 90)

plt.xlabel('DATE TIME')

plt.ylabel('AMBIENT/MODULE TEMPERATURE')

plt.grid()

plt.legend()

plt.show()
plt.subplots(1,1, figsize=(20,10))

plt.title("Daily yield of both plants across different dates")

plt.plot(df1.DATE_TIME, df1.DAILY_YIELD.rolling(window=20).mean(), label='Plant 1 Daily Yield', c='g')

plt.plot(df3.DATE_TIME, df3.DAILY_YIELD.rolling(window=20).mean(), label='Plant 2 Daily Yield')

plt.xlabel('DATE TIME')

plt.ylabel('DAILY YIELD')

plt.grid()

plt.legend()

plt.show()
TOTAL = df1.groupby('SOURCE_KEY')['TOTAL_YIELD'].max()

inv_lst = df1['SOURCE_KEY'].unique()

plt.subplots(1,1, figsize=(20,10))

plt.bar(inv_lst,TOTAL, label="PLANT 1 TOTAL YIELD")

plt.xticks(rotation=90)

plt.xlabel('INVERTORS')

plt.ylabel('TOTAL YIELD')

plt.grid()

plt.legend()

plt.show()
TOTAL = df3.groupby('SOURCE_KEY')['TOTAL_YIELD'].max()

inv_lst = df3['SOURCE_KEY'].unique()

plt.subplots(1,1, figsize=(20,10))

plt.bar(inv_lst,TOTAL, label="PLANT 2 TOTAL YIELD")

plt.xticks(rotation=90)

plt.xlabel('INVERTORS')

plt.ylabel('TOTAL YIELD')

plt.grid()

plt.legend()

plt.show()
dfgen1 = pd.merge(df1, df2, on =['DATE_TIME', 'DATE', 'TIME'], how='left')

irr = dfgen1.groupby('DATE')['IRRADIATION'].sum()

date = dfgen1['DATE'].unique()

plt.subplots(1,1, figsize=(20,10))

plt.barh(date, irr, color = 'black')

plt.xlabel('IRRADIATION')

plt.ylabel('DATE')

plt.title('Irradiation vs DateTime Plant1')

plt.show()
dfgen2 = pd.merge(df3, df4, on =['DATE_TIME', 'DATE', 'TIME'], how='left')

irr = dfgen2.groupby('DATE')['IRRADIATION'].sum()

date = dfgen2['DATE'].unique()

plt.subplots(1,1, figsize=(20,10))

plt.barh(date, irr, color = 'black')

plt.xlabel('IRRADIATION')

plt.ylabel('DATE')

plt.title('Irradiation vs DateTime Plant2')

plt.show()
r_left = pd.merge(df1, df2, on ='DATE_TIME', how='left')

plt.subplots(1,1, figsize=(20,10))

plt.title('IRRADIATION VS DC POWER')

plt.plot(r_left['IRRADIATION'], r_left['DC_POWER'], c='c', marker ='o', linestyle='', alpha = 0.1, label ='DC POWER')

plt.legend()

plt.grid()

plt.xlabel('IRRADIATION')

plt.ylabel('DC POWER')

plt.show()
plt.subplots(1,1, figsize=(20,10))

plt.title('IRRADIATION VS AC POWER')

plt.plot(r_left['IRRADIATION'], r_left['AC_POWER'], c='y', marker ='o', linestyle='', alpha = 0.1, label ='AC POWER')

plt.legend()

plt.grid()

plt.xlabel('IRRADIATION')

plt.ylabel('AC POWER')

plt.show()
dates = r_left['DATE_x'].unique()

plt.subplots(1,1, figsize=(20,10))

plt.title('Plant 1')

for date in dates:

    data = df2[df2['DATE']==date][df2['IRRADIATION']>0]

    plt.plot(data['AMBIENT_TEMPERATURE'], data['MODULE_TEMPERATURE'], 

             marker = 'o', 

             linestyle ='', 

             alpha = 0.5, 

             ms= 6,

             label = pd.to_datetime(date,format = '%Y-%m-%d').date()

            )

plt.legend()

plt.show()
r_left = pd.merge(df3, df4, on ='DATE_TIME', how='left')

dates = r_left['DATE_x'].unique()

plt.subplots(1,1, figsize=(20,10))

plt.title('Plant 2')

for date in dates:

    data = df4[df4['DATE']==date][df4['IRRADIATION']>0]

    plt.plot(data['AMBIENT_TEMPERATURE'], data['MODULE_TEMPERATURE'], 

             marker = 'o', 

             linestyle ='', 

             alpha = 0.5, 

             ms= 6,

             label = pd.to_datetime(date,format = '%Y-%m-%d').date()

            )

plt.legend()

plt.show()