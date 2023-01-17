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
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_pgen1
df_pgen2
df_wsen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_wsen2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_wsen1
df_wsen2
df_wsen1.info()
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1.info()
df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
df_pgen2['DATE'] = df_pgen2['DATE_TIME'].apply(lambda x:x.date())
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].apply(lambda x:x.time())
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')
df_pgen2.info()
df_pgen2

df_wsen1['DATE_TIME'] = pd.to_datetime(df_wsen1['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
df_wsen1['DATE'] = df_wsen1['DATE_TIME'].apply(lambda x:x.date())
df_wsen1['TIME'] = df_wsen1['DATE_TIME'].apply(lambda x:x.time())
df_wsen1['DATE'] = pd.to_datetime(df_wsen1['DATE'],format = '%Y-%m-%d')
df_wsen1.info()
df_wsen2['DATE_TIME'] = pd.to_datetime(df_wsen2['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')
df_wsen2['DATE'] = df_wsen2['DATE_TIME'].apply(lambda x:x.date())
df_wsen2['TIME'] = df_wsen2['DATE_TIME'].apply(lambda x:x.time())
df_wsen2['DATE'] = pd.to_datetime(df_wsen2['DATE'],format = '%Y-%m-%d')
df_wsen2.info()
df_wsen1
#What is the mean value of daily yield?
a = df_pgen1['DAILY_YIELD'].mean()
print("MEAN DAILY YIELD OF PLANT1 : ",a)
b = df_pgen2['DAILY_YIELD'].mean()
print("MEAN DAILY YIELD OF PLANT1 : ",b)
c = (a+b)/2
print("\nMEAN DAILY YIELD: ",c)
#What is the total irradiation per day?
a = df_wsen1.groupby(['DATE']).sum()
print("IRRADIATION PER DAY OF PLANT1: \n",a['IRRADIATION'])
b = df_wsen2.groupby(['DATE']).sum()
print("IRRADIATION PER DAY OF PLANT2: ",b['IRRADIATION'])
#What is the max ambient and module temperature?
print("MAX AMBIENT TEMPERATURE OF PLANT1 : ",df_wsen1['AMBIENT_TEMPERATURE'].max())
print("MAX AMBIENT TEMPERATURE OF PLANT2 : ",df_wsen2['AMBIENT_TEMPERATURE'].max())
print("\nMAX MODULE TEMPERATURE OF PLANT1: ",df_wsen1['MODULE_TEMPERATURE'].max())
print("MAX MODULE TEMPERATURE OF PLANT2  : ",df_wsen2['MODULE_TEMPERATURE'].max())
#How many inverters are there for each plant?
print("NUMBER OF INVERTERS IN PLANT1: ",df_pgen1['SOURCE_KEY'].nunique())
print("NUMBER OF INVERTERS IN PLANT2: ",df_pgen2['SOURCE_KEY'].nunique())
print("\nTOTAL INVERTERS: ",df_pgen1['SOURCE_KEY'].nunique()+df_pgen2['SOURCE_KEY'].nunique())
#What is the maximum/minimum amount of DC/AC Power generated in a time interval/day?
a = df_pgen1.groupby(['DATE'])
print("\nMAXIMUM AMOUNT OF DC IN A DAY GEN BY PLANT1: ",a['DC_POWER'].max())
a1 = df_pgen1.groupby(['DATE'])
print("\nMINIMUM AMOUNT OF DC IN A DAY GEN BY PLANT1: ",a['DC_POWER'].min())
b = df_pgen2.groupby(['DATE'])
print("\nMAXIMUM AMOUNT OF DC IN A DAY GEN BY PLANT2: ",b['DC_POWER'].max())
b1 = df_pgen2.groupby(['DATE'])
print("\nMINIMUM AMOUNT OF DC IN A DAY GEN BY PLANT2: ",b['DC_POWER'].min())
#Which inverter (source_key) has produced maximum DC/AC power?
a = df_pgen1['DC_POWER'].max()       #HOLDS THE MAX DCPOWER GENERATED
print("\n INVERTER PRODUCING MAX DCPOWER IN PLANT1: ")
print(df_pgen1[['SOURCE_KEY','DC_POWER']][df_pgen1['DC_POWER'] == a])
a2 = df_pgen1['AC_POWER'].max()       #HOLDS THE MAX DCPOWER GENERATED
print("\n INVERTER PRODUCING MAX ACPOWER IN PLANT1: ")
print(df_pgen1[['SOURCE_KEY','AC_POWER']][df_pgen1['AC_POWER'] == a2])
a1 = df_pgen2['DC_POWER'].max()       #HOLDS THE MAX DCPOWER GENERATED
print("\n INVERTER PRODUCING MAX DCPOWER IN PLANT2: ")
print(df_pgen2[['SOURCE_KEY','DC_POWER']][df_pgen2['DC_POWER'] == a1])
a12 = df_pgen2['AC_POWER'].max()       #HOLDS THE MAX DCPOWER GENERATED
print("\n INVERTER PRODUCING MAX ACPOWER IN PLANT2: ")
print(df_pgen2[['SOURCE_KEY','AC_POWER']][df_pgen2['AC_POWER'] == a12])
#Rank the inverters based on the DC/AC power they produce.
a = df_pgen1.groupby(['SOURCE_KEY']).count()
b = a.iloc[:,[2,3]].rank()
print("\nPLANT1 INVERTERS RANKING BASED ON DC/AC POWER: \n",b)
a1 = df_pgen2.groupby(['SOURCE_KEY']).count()
b1 = a1.iloc[:,[2,3]].rank()
print("\nPLANT2 INVERTERS RANKING BASED ON DC/AC POWER: \n",b1)
# Is there any missing data?
#To see if there's any data missing in the datasets, we must check if there's any place void in the whole dataset
df_pgen1.isnull()
df_pgen2.isnull()
df_wsen1.isnull()
df_wsen2.isnull()
