# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# task1 reading csv files

plant1_gd_df = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")

plant1_wsd_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

plant2_gd_df = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv")

plant2_wsd_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
print(plant1_gd_df.dtypes)

plant1_gd_df.head()
#plant 1 generation data

plant1_gd_df.describe()
print(plant1_wsd_df.dtypes)

plant1_wsd_df.head()
# Plant 1 weather sensor data

plant1_wsd_df.describe()
print(plant2_gd_df.dtypes)

plant2_gd_df.head()
plant2_gd_df.describe()
print(plant2_wsd_df.dtypes)

plant2_wsd_df.head()
plant2_wsd_df.describe()
# Converting date_time column to datetime type



plant1_gd_df['DATE_TIME'] = pd.to_datetime(plant1_gd_df['DATE_TIME'], format = '%d-%m-%Y %H:%M')

plant1_wsd_df['DATE_TIME'] = pd.to_datetime(plant1_wsd_df['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

plant2_gd_df['DATE_TIME'] = pd.to_datetime(plant2_gd_df['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')

plant2_wsd_df['DATE_TIME'] = pd.to_datetime(plant2_wsd_df['DATE_TIME'], format = '%Y-%m-%d %H:%M:%S')



# Plant 1 generation data

print("Plant 1 generation data")

print(plant1_gd_df.isna().sum())
# Plant 1 weather sensor data

print("Plant 1 weather sensor data")

print(plant1_wsd_df.isna().sum())
# Plant 2 generation data

print("Plant 2 generation data")

print(plant1_gd_df.isna().sum())
# Plant 2 weather sensor data

print("Plant 2 weather sensor data")

print(plant1_wsd_df.isna().sum())
print("Mean of daily yield")

print("Plant 1 :", plant1_gd_df['DAILY_YIELD'].mean())

print("Plant 2 :", plant2_gd_df['DAILY_YIELD'].mean())
# creating column for day

plant1_wsd_df['DAY'] = plant1_wsd_df['DATE_TIME'].dt.date

plant2_wsd_df['DAY'] = plant2_wsd_df['DATE_TIME'].dt.date
plant1_irradiation = plant1_wsd_df.groupby('DAY')['IRRADIATION'].sum()

plant2_irradiation = plant2_wsd_df.groupby('DAY')['IRRADIATION'].sum()



fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))

plant1_irradiation.plot(ax = ax[0], style = 'o--')

plant2_irradiation.plot(ax = ax[1], style = 'o--')



ax[0].set_title("Irradiation per day in plant 1")

ax[1].set_title("Irradiation per day in plant 1")

ax[0].set_ylabel("Irradiation")

ax[1].set_ylabel("Irradiation")

ax[0].tick_params(labelrotation=45)

ax[1].tick_params(labelrotation=45)

print("Plant 1 total irradiation per day")

print(plant1_irradiation)

print("\nPlant 2 total irradiation per day")

print(plant2_irradiation)
print("Max Ambient and Module temperature")

print("Plant 1 MAX Ambient Temperature :", plant1_wsd_df['AMBIENT_TEMPERATURE'].max())

print("Plant 1 MAX Module Temperature :", plant1_wsd_df['MODULE_TEMPERATURE'].max())

print("\nPlant 2 MAX Ambient Temperature :", plant2_wsd_df['AMBIENT_TEMPERATURE'].max())

print("Plant 2 MAX Module Temperature :", plant2_wsd_df['MODULE_TEMPERATURE'].max())
print("Number of Inverters in each plant \n Plant1:")

print(plant1_gd_df.groupby('PLANT_ID')['SOURCE_KEY'].nunique().reset_index(name = 'Number of Inverters'))

print("\n Plant 2:")

print(plant2_gd_df.groupby('PLANT_ID')['SOURCE_KEY'].nunique().reset_index(name = 'Number of Inverters'))
plant1_gd_df["TIME"] = plant1_gd_df["DATE_TIME"].dt.time

plant2_gd_df["TIME"] = plant2_gd_df["DATE_TIME"].dt.time

fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))

plant1_gd_df.groupby(["TIME"])[["DC_POWER","AC_POWER"]].max().reset_index().plot(x='TIME', style = 'o', ax = ax[0], label = 'Plant 1')

plant2_gd_df.groupby(["TIME"])[["DC_POWER","AC_POWER"]].max().reset_index().plot(x='TIME', style = 'o', ax = ax[1], label = 'Plant 2')

ax[0].set_title('Plant 1 maximum DC/AC power in time interval')

ax[1].set_title('Plant 2 maximum DC/AC power in time interval')

ax[0].set_ylabel('Power Generated')

ax[1].set_ylabel('Power Generated')
fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))

plant1_gd_df.groupby(["TIME"])[["DC_POWER","AC_POWER"]].min().reset_index().plot(x='TIME', style = 'o', ax = ax[0], label = 'Plant 1')

plant2_gd_df.groupby(["TIME"])[["DC_POWER","AC_POWER"]].min().reset_index().plot(x='TIME', style = 'o', ax = ax[1], label = 'Plant 2')

ax[0].set_title('Plant 1 minimum DC/AC power in time interval')

ax[1].set_title('Plant 2 minimum DC/AC power in time interval')

ax[0].set_ylabel('Power Generated')

ax[1].set_ylabel('Power Generated')
plant1_gd_df["DAY"] = plant1_gd_df["DATE_TIME"].dt.date

plant2_gd_df["DAY"] = plant2_gd_df["DATE_TIME"].dt.date





fig,ax = plt.subplots(ncols=2,nrows=2,dpi=100,figsize=(20,10))

plant1_gd_df.groupby(["DAY"])[["DC_POWER","AC_POWER"]].min().reset_index().plot(x='DAY', style = 'o', ax = ax[0,0], label = 'Plant 1')

plant2_gd_df.groupby(["DAY"])[["DC_POWER","AC_POWER"]].min().reset_index().plot(x='DAY', style = 'o', ax = ax[0,1], label = 'Plant 2')

plant1_gd_df.groupby(["DAY"])[["DC_POWER","AC_POWER"]].max().reset_index().plot(x='DAY', style = 'o', ax = ax[1,0], label = 'Plant 1')

plant2_gd_df.groupby(["DAY"])[["DC_POWER","AC_POWER"]].max().reset_index().plot(x='DAY', style = 'o', ax = ax[1,1], label = 'Plant 2')



ax[0,0].set_title('Plant 1 minimum DC/AC power in Day')

ax[0,1].set_title('Plant 2 minimum DC/AC power in Day')

ax[1,0].set_title('Plant 1 maximum DC/AC power in Day')

ax[1,1].set_title('Plant 2 maximum DC/AC power in Day')



ax[0,0].set_ylabel('Power Generated')

ax[0,1].set_ylabel('Power Generated')

ax[1,0].set_ylabel('Power Generated')

ax[1,1].set_ylabel('Power Generated')



ax[0,0].tick_params(labelrotation=45)

ax[0,1].tick_params(labelrotation=45)

ax[1,0].tick_params(labelrotation=45)

ax[1,1].tick_params(labelrotation=45)



fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))

plant1 = plant1_gd_df.groupby(["SOURCE_KEY"])[["DC_POWER","AC_POWER"]].max().reset_index() 

plant2 = plant2_gd_df.groupby(["SOURCE_KEY"])[["DC_POWER","AC_POWER"]].max().reset_index()

plant1.plot(kind = 'bar', x='SOURCE_KEY', ax = ax[0], label = 'Plant 1')

plant2.plot(kind = 'bar', x='SOURCE_KEY', ax = ax[1], label = 'Plant 2')

ax[0].set_title('Plant 1 maximum DC/AC power generated by inverter')

ax[1].set_title('Plant 2 Maximum DC/AC power generated by inverter')

ax[0].set_ylabel('Power Generated')

ax[1].set_ylabel('Power Generated')

print("Plant 1:")

print(plant1)

print("\nPlant 2:")

print(plant2)
fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))



plant1_gd_df.groupby(["SOURCE_KEY"])["DC_POWER"].sum().reset_index().sort_values('DC_POWER', ascending = False).plot(kind = 'bar', x = 'SOURCE_KEY', ax = ax[0])

plant2_gd_df.groupby(["SOURCE_KEY"])["DC_POWER"].sum().reset_index().sort_values('DC_POWER', ascending = False).plot(kind = 'bar', x = 'SOURCE_KEY', ax = ax[1])



ax[0].set_title('Plant 1 Inverter rank based sum of DC power')

ax[1].set_title('Plant 1 Inverter rank based sum of DC power')

ax[0].set_ylabel('Total Power Generated')

ax[1].set_ylabel('Total Power Generated')

fig,ax = plt.subplots(ncols=2,nrows=1,dpi=100,figsize=(20,5))



plant1_gd_df.groupby(["SOURCE_KEY"])["AC_POWER"].sum().reset_index().sort_values('AC_POWER', ascending = False).plot(kind = 'bar', x = 'SOURCE_KEY', ax = ax[0])

plant2_gd_df.groupby(["SOURCE_KEY"])["AC_POWER"].sum().reset_index().sort_values('AC_POWER', ascending = False).plot(kind = 'bar', x = 'SOURCE_KEY', ax = ax[1])



ax[0].set_title('Plant 1 Inverter rank based sum of AC power')

ax[1].set_title('Plant 1 Inverter rank based sum of AC power')

ax[0].set_ylabel('Total Power Generated')

ax[1].set_ylabel('Total Power Generated')
