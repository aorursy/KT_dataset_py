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
# Load the data from the CSV files
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_wsen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
print("Power Generation Data:\n")
df_pgen1
print("Weather Sensor Data:\n")
df_wsen1
df_pgen1.info()
df_wsen1.info()
df_pgen1.describe()
df_wsen1.describe()
df_pgen1.nunique()
df_wsen1.nunique()
df_pgen1['DATE_TIME'] = pd.to_datetime(df_pgen1['DATE_TIME'],format = '%d-%m-%Y %H:%M')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].dt.date
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].dt.time
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_pgen1.info()
df_wsen1['DATE_TIME'] = pd.to_datetime(df_wsen1['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_wsen1['DATE'] = df_wsen1['DATE_TIME'].dt.date
df_wsen1['DATE'] = pd.to_datetime(df_wsen1['DATE'],format = '%Y-%m-%d')
df_wsen1['TIME'] = df_wsen1['DATE_TIME'].dt.time
df_wsen1['HOUR'] = pd.to_datetime(df_wsen1['TIME'],format='%H:%M:%S').dt.hour
df_wsen1['MINUTES'] = pd.to_datetime(df_wsen1['TIME'],format='%H:%M:%S').dt.minute
df_wsen1.info()
# 1. What is the mean value of daily yield?
print("1. The mean value of the daily yield is:", df_pgen1['DAILY_YIELD'].mean())

# 2. What is the total irradiation per day?
print("\n2. The total irradiation per day is:\n",df_wsen1.groupby(['DATE'])['IRRADIATION'].sum())

# 3. What is the max ambient and module temperature?
print("\n3a. The max ambient temperature is:", df_wsen1['AMBIENT_TEMPERATURE'].max())
print("3b. The max module temperature is:", df_wsen1['MODULE_TEMPERATURE'].max())

# 4. How many inverters are there for each plant?
print("\n4. The number of inverters in the plant are:", len(df_pgen1['SOURCE_KEY'].unique()))

# 5. What is the maximum/minimum amount of DC/AC Power generated in a time interval/day?
daily_DC = df_pgen1.groupby(['DATE'])['DC_POWER'].sum()
daily_AC = df_pgen1.groupby(['DATE'])['AC_POWER'].sum()
print("\n5a. The maximum amount of DC Power generated in a day is:", daily_DC.max())
print("5b. The minimum amount of DC Power generated in a day is:", daily_DC.min())
print("5c. The maximum amount of AC Power generated in a day is:", daily_AC.max())
print("5d. The minimum amount of AC Power generated in a day is:", daily_AC.min())

#6. Which inverter (source_key) has produced maximum DC/AC power?
max_dc = df_pgen1['DC_POWER'].max()
max_ac = df_pgen1['AC_POWER'].max()
dc_inverter = df_pgen1['SOURCE_KEY'].values[df_pgen1['DC_POWER'].argmax()]
ac_inverter = df_pgen1['SOURCE_KEY'].values[df_pgen1['AC_POWER'].argmax()]
print("\n6a. The maximum DC Power is", max_dc, "and the inverter that produced it is", dc_inverter)
print("6b. The maximum AC Power is", max_ac, "and the inverter that produced it is", ac_inverter)

# 7. Rank the inverters based on the DC/AC power they produce
print("\n7a. Ranking based on DC Power:\n", df_pgen1.groupby('SOURCE_KEY').max().sort_values(by=['DC_POWER'], ascending=False)['DC_POWER'])
print("\n7b. Ranking based on AC Power:\n", df_pgen1.groupby('SOURCE_KEY').max().sort_values(by=['AC_POWER'], ascending = False)['AC_POWER'])

# 8. Is there any missing data?
ideal_values = 34*24*4
values = df_pgen1['SOURCE_KEY'].value_counts()
missing = (ideal_values)*22 - values.sum()
print("\n8. Ideally there should have been", ideal_values, "readings per inverter but the distribution per inverter is:\n", values)
print("\nSo there are", missing, "values missing from the dataset.")
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_wsen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
print("Power Generation Data:\n")
df_pgen2
print("Weather Sensor Data:\n")
df_wsen2
df_pgen2.info()
df_wsen2.info()
df_pgen2.describe()
df_wsen2.describe()
df_pgen2.nunique()
df_wsen2.nunique()
df_pgen2['DATE_TIME'] = pd.to_datetime(df_pgen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_pgen2['DATE'] = df_pgen2['DATE_TIME'].dt.date
df_pgen2['DATE'] = pd.to_datetime(df_pgen2['DATE'],format = '%Y-%m-%d')
df_pgen2['TIME'] = df_pgen2['DATE_TIME'].dt.time
df_pgen2['HOUR'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.hour
df_pgen2['MINUTES'] = pd.to_datetime(df_pgen2['TIME'],format='%H:%M:%S').dt.minute
df_pgen2.info()
df_wsen2['DATE_TIME'] = pd.to_datetime(df_wsen2['DATE_TIME'],format = '%Y-%m-%d %H:%M')
df_wsen2['DATE'] = df_wsen2['DATE_TIME'].dt.date
df_wsen2['DATE'] = pd.to_datetime(df_wsen2['DATE'],format = '%Y-%m-%d')
df_wsen2['TIME'] = df_wsen2['DATE_TIME'].dt.time
df_wsen2['HOUR'] = pd.to_datetime(df_wsen2['TIME'],format='%H:%M:%S').dt.hour
df_wsen2['MINUTES'] = pd.to_datetime(df_wsen2['TIME'],format='%H:%M:%S').dt.minute
df_wsen2.info()
# 1. What is the mean value of daily yield?
print("1. The mean value of the daily yield is:", df_pgen2['DAILY_YIELD'].mean())

# 2. What is the total irradiation per day?
print("\n2. The total irradiation per day is:\n",df_wsen2.groupby(['DATE'])['IRRADIATION'].sum())

# 3. What is the max ambient and module temperature?
print("\n3a. The max ambient temperature is:", df_wsen2['AMBIENT_TEMPERATURE'].max())
print("3b. The max module temperature is:", df_wsen2['MODULE_TEMPERATURE'].max())

# 4. How many inverters are there for each plant?
print("\n4. The number of inverters in the plant are:", len(df_pgen2['SOURCE_KEY'].unique()))

# 5. What is the maximum/minimum amount of DC/AC Power generated in a time interval/day?
daily_DC = df_pgen2.groupby(['DATE'])['DC_POWER'].sum()
daily_AC = df_pgen2.groupby(['DATE'])['AC_POWER'].sum()
print("\n5a. The maximum amount of DC Power generated in a day is:", daily_DC.max())
print("5b. The minimum amount of DC Power generated in a day is:", daily_DC.min())
print("5c. The maximum amount of AC Power generated in a day is:", daily_AC.max())
print("5d. The minimum amount of AC Power generated in a day is:", daily_AC.min())

#6. Which inverter (source_key) has produced maximum DC/AC power?
max_dc = df_pgen2['DC_POWER'].max()
max_ac = df_pgen2['AC_POWER'].max()
dc_inverter = df_pgen2['SOURCE_KEY'].values[df_pgen2['DC_POWER'].argmax()]
ac_inverter = df_pgen2['SOURCE_KEY'].values[df_pgen2['AC_POWER'].argmax()]
print("\n6a. The maximum DC Power is", max_dc, "and the inverter that produced it is", dc_inverter)
print("6b. The maximum AC Power is", max_ac, "and the inverter that produced it is", ac_inverter)

# 7. Rank the inverters based on the DC/AC power they produce
print("\n7a. Ranking based on DC Power:\n", df_pgen2.groupby('SOURCE_KEY').max().sort_values(by=['DC_POWER'], ascending=False)['DC_POWER'])
print("\n7b. Ranking based on AC Power:\n", df_pgen2.groupby('SOURCE_KEY').max().sort_values(by=['AC_POWER'], ascending = False)['AC_POWER'])

# 8. Is there any missing data?
ideal_values = 34*24*4
values = df_pgen2['SOURCE_KEY'].value_counts()
missing = (ideal_values)*22 - values.sum()
print("\n8. Ideally there should have been", ideal_values, "readings per inverter but the distribution per inverter is:\n", values)
print("\nSo there are", missing, "values missing from the dataset.")