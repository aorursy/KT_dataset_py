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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
Plant_1_Generation_Data = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

Plant_1_Weather_Sensor_Data = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

Plant_2_Weather_Sensor_Data = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

Plant_2_Generation_Data = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
print(('-'*50)+'PLANT 1 INFORMATION'+('-'*50))

print('\n')

print(('-')*50+'Generation Data'+('-')*50)

print('\n')

print(Plant_1_Generation_Data.columns)

print('\n')

print(Plant_1_Generation_Data.head())

print('\n')

print(Plant_1_Generation_Data.describe())

print('\n')

print(Plant_1_Generation_Data.info())

print('\n\n')

print(('-')*50+'Weather Sensor Data'+('-')*50)

print('\n')

print(Plant_1_Weather_Sensor_Data.columns)

print('\n')

print(Plant_1_Weather_Sensor_Data.head())

print('\n')

print(Plant_1_Weather_Sensor_Data.describe())

print('\n')

print(Plant_1_Weather_Sensor_Data.info())
print(('-'*50)+'PLANT 2 INFORMATION'+('-'*50))

print('\n')

print(('-')*50+'Generation Data'+('-')*50)

print('\n')

print(Plant_2_Generation_Data.columns)

print('\n')

print(Plant_2_Generation_Data.head())

print('\n')

print(Plant_2_Generation_Data.describe())

print('\n')

print(Plant_2_Generation_Data.info())

print('\n\n')

print(('-')*50+'Weather Sensor Data'+('-')*50)

print('\n')

print(Plant_2_Weather_Sensor_Data.columns)

print('\n')

print(Plant_2_Weather_Sensor_Data.head())

print('\n')

print(Plant_2_Weather_Sensor_Data.describe())

print('\n')

print(Plant_2_Weather_Sensor_Data.info())
missing_value_count_plant1_generation_data = Plant_1_Generation_Data.isnull().sum()

missing_value_count_plant1_weather_data = Plant_1_Weather_Sensor_Data.isnull().sum()

missing_value_count_plant2_generation_data = Plant_2_Generation_Data.isnull().sum()

missing_value_count_plant2_weather_data = Plant_2_Weather_Sensor_Data.isnull().sum()
print(missing_value_count_plant1_generation_data)

print(missing_value_count_plant1_weather_data)

print(missing_value_count_plant2_generation_data)

print(missing_value_count_plant2_weather_data)
print(('-'*20),'Plant_1_Generation_Data',('-'*20))

print(Plant_1_Generation_Data['DATE_TIME'].head())

print('\n')

print(('-'*20)+'-'*len('Plant_1_Generation_Data')+('-'*22))

print(Plant_1_Generation_Data['DATE_TIME'].tail())

print('\n\n')

print(('-'*20),'Plant_2_Generation_Data',('-'*20))

print(Plant_2_Generation_Data['DATE_TIME'].head())

print('\n')

print(('-'*20)+'-'*len('Plant_2_Generation_Data')+('-'*22))

print(Plant_2_Generation_Data['DATE_TIME'].tail())

print('\n\n')

print(('-'*20),'Plant_1_Weather_Sensor_Data',('-'*20))

print(Plant_1_Weather_Sensor_Data['DATE_TIME'].head())

print('\n')

print(('-'*20)+'-'*len('Plant_1_Weather_Sensor_Data')+('-'*22))

print(Plant_1_Weather_Sensor_Data['DATE_TIME'].tail())

print('\n\n')

print(('-'*20),'Plant_2_Weather_Sensor_Data',('-'*20))

print(Plant_2_Weather_Sensor_Data['DATE_TIME'].head())

print('\n')

print(('-'*20)+'-'*len('Plant_2_Weather_Sensor_Data')+('-'*22))

print(Plant_2_Weather_Sensor_Data['DATE_TIME'].tail())
Plant_1_Generation_Data['DATE_TIME_PARSED'] = pd.to_datetime(Plant_1_Generation_Data['DATE_TIME'],format = "%d-%m-%Y %H:%M")

Plant_1_Weather_Sensor_Data['DATE_TIME_PARSED'] = pd.to_datetime(Plant_1_Weather_Sensor_Data['DATE_TIME'],format = "%Y-%m-%d %H:%M:%S")

Plant_2_Generation_Data['DATE_TIME_PARSED'] = pd.to_datetime(Plant_2_Generation_Data['DATE_TIME'],format = "%Y-%m-%d %H:%M:%S")

Plant_2_Weather_Sensor_Data['DATE_TIME_PARSED'] = pd.to_datetime(Plant_2_Weather_Sensor_Data['DATE_TIME'],format = "%Y-%m-%d %H:%M:%S")
print(('*')*30+'DATE_TIME_PARSED'+('*')*30)

print(('-'*20),'Plant_1_Generation_Data',('-'*20))

print(Plant_1_Generation_Data['DATE_TIME_PARSED'].head())

print('\n\n')

print(('-'*20),'Plant_1_Weather_Sensor_Data',('-'*20))

print(Plant_1_Weather_Sensor_Data['DATE_TIME_PARSED'].head())

print('\n\n')

print(('-'*20),'Plant_2_Generation_Data',('-'*20))

print(Plant_2_Generation_Data['DATE_TIME_PARSED'].head())

print('\n\n')

print(('-'*20),'Plant_2_Weather_Sensor_Data',('-'*20))

print(Plant_2_Weather_Sensor_Data['DATE_TIME_PARSED'].head())
print('Daily yield by Plant 1:',Plant_1_Generation_Data.DAILY_YIELD.mean())

print('Daily yield by Plant 2:',Plant_2_Generation_Data.DAILY_YIELD.mean())
plant_1_irradiation_per_day = Plant_1_Weather_Sensor_Data.groupby(Plant_1_Weather_Sensor_Data.DATE_TIME_PARSED.dt.date)['IRRADIATION'].sum()

plant_2_irradiation_per_day = Plant_2_Weather_Sensor_Data.groupby(Plant_2_Weather_Sensor_Data.DATE_TIME_PARSED.dt.date)['IRRADIATION'].sum()
print('PLANT 1 IRRADIATION PER DAY')

print('\n')

print(plant_1_irradiation_per_day)

print('\n\n')

print('PLANT 2 IRRADIATION PER DAY')

print('\n')

print(plant_2_irradiation_per_day)
print('PLANT 1 IRRADIATION PER DAY')

plt.figure(figsize= (13,13))

plant_1_irradiation_per_day.plot(kind = 'bar')

plt.show()

print('PLANT 2 IRRADIATION PER DAY')

plt.figure(figsize= (13,13))

plant_2_irradiation_per_day.plot(kind = 'bar')

plt.show()
plant_1_max_ambient_temperature = Plant_1_Weather_Sensor_Data.AMBIENT_TEMPERATURE.max()

plant_1_max_module_temperature = Plant_1_Weather_Sensor_Data.MODULE_TEMPERATURE.max()

plant_2_max_ambient_temperature = Plant_2_Weather_Sensor_Data.AMBIENT_TEMPERATURE.max()

plant_2_max_module_temperature = Plant_2_Weather_Sensor_Data.MODULE_TEMPERATURE.max()
print('Max ambient Temperature by Plant 1:',plant_1_max_ambient_temperature)

print('Max ambient Temperature by Plant 2:',plant_2_max_ambient_temperature)

print('Max module Temperature by Plant 1:',plant_1_max_module_temperature)

print('Max module Temperature by Plant 2:',plant_2_max_module_temperature)
inverters_count_for_plant_1 = len(Plant_1_Generation_Data.SOURCE_KEY.unique())

inverters_count_for_plant_2 = len(Plant_2_Generation_Data.SOURCE_KEY.unique())
print('INVERTERS COUNT FOR PLANT 1:',inverters_count_for_plant_1)

print('INVERTERS COUNT FOR PLANT 2:',inverters_count_for_plant_2)
MAX_DC_Power_per_day_by_plant_1 = Plant_1_Generation_Data.groupby(Plant_1_Generation_Data.DATE_TIME_PARSED.dt.date)['DC_POWER'].max()

MIN_DC_Power_per_day_by_plant_1 = Plant_1_Generation_Data.groupby(Plant_1_Generation_Data.DATE_TIME_PARSED.dt.date)['DC_POWER'].min()



MAX_DC_Power_per_day_by_plant_2 = Plant_2_Generation_Data.groupby(Plant_2_Generation_Data.DATE_TIME_PARSED.dt.date)['DC_POWER'].max()

MIN_DC_Power_per_day_by_plant_2 = Plant_2_Generation_Data.groupby(Plant_2_Generation_Data.DATE_TIME_PARSED.dt.date)['DC_POWER'].min()



MAX_AC_Power_per_day_by_plant_1 = Plant_1_Generation_Data.groupby(Plant_1_Generation_Data.DATE_TIME_PARSED.dt.date)['AC_POWER'].max()

MIN_AC_Power_per_day_by_plant_1 = Plant_1_Generation_Data.groupby(Plant_1_Generation_Data.DATE_TIME_PARSED.dt.date)['AC_POWER'].min()



MAX_AC_Power_per_day_by_plant_2 = Plant_2_Generation_Data.groupby(Plant_2_Generation_Data.DATE_TIME_PARSED.dt.date)['AC_POWER'].max()

MIN_AC_Power_per_day_by_plant_2 = Plant_2_Generation_Data.groupby(Plant_2_Generation_Data.DATE_TIME_PARSED.dt.date)['AC_POWER'].min()
MAX_DC_Power_per_day_by_plant_1.plot(kind = 'bar',title = 'MAX_DC_POWER PER DAY BY PLANT 1')

plt.show()

MAX_DC_Power_per_day_by_plant_2.plot(kind = 'bar',title = 'MAX_DC_POWER PER DAY BY PLANT 2')

plt.show()

MIN_DC_Power_per_day_by_plant_1.plot(kind = 'bar',title = 'MIN_DC_POWER PER DAY BY PLANT 1')

plt.show()

MAX_DC_Power_per_day_by_plant_2.plot(kind = 'bar',title = 'MIN_DC_POWER PER DAY BY PLANT 2')

plt.show()

MAX_AC_Power_per_day_by_plant_1.plot(kind = 'bar',title = 'MAX_AC_POWER PER DAY BY PLANT 1')

plt.show()

MAX_AC_Power_per_day_by_plant_2.plot(kind = 'bar',title = 'MAX_AC_POWER PER DAY BY PLANT 2')

plt.show()

MIN_AC_Power_per_day_by_plant_1.plot(kind = 'bar',title = 'MIN_AC_POWER PER DAY BY PLANT 1')

plt.show()

MAX_AC_Power_per_day_by_plant_2.plot(kind = 'bar',title = 'MIN_AC_POWER PER DAY BY PLANT 2')

plt.show()
max_DC_power_source_plant_1 = Plant_1_Generation_Data.loc[Plant_1_Generation_Data['DC_POWER'].idxmax()]['SOURCE_KEY']

max_AC_power_source_plant_1 = Plant_1_Generation_Data.loc[Plant_1_Generation_Data['AC_POWER'].idxmax()]['SOURCE_KEY']

max_DC_power_source_plant_2 = Plant_2_Generation_Data.loc[Plant_2_Generation_Data['DC_POWER'].idxmax()]['SOURCE_KEY']

max_AC_power_source_plant_2 = Plant_2_Generation_Data.loc[Plant_2_Generation_Data['AC_POWER'].idxmax()]['SOURCE_KEY']

print('MAX DC POWER GENERATED BY PLANT 1 THROUGH SOURCE KEY:',max_DC_power_source_plant_1)

print('MAX AC POWER GENERATED BY PLANT 1 THROUGH SOURCE KEY:',max_AC_power_source_plant_1)

print('MAX DC POWER GENERATED BY PLANT 2 THROUGH SOURCE KEY:',max_DC_power_source_plant_2)

print('MAX AC POWER GENERATED BY PLANT 2 THROUGH SOURCE KEY:',max_AC_power_source_plant_2)
DC_POWER_per_source_key_plant_1 = Plant_1_Generation_Data.groupby(Plant_1_Generation_Data.SOURCE_KEY)['DC_POWER'].sum()

DC_POWER_per_source_key_plant_2 = Plant_2_Generation_Data.groupby(Plant_2_Generation_Data.SOURCE_KEY)['DC_POWER'].sum()
print(DC_POWER_per_source_key_plant_1.rank().sort_values())
print(DC_POWER_per_source_key_plant_2.rank().sort_values())