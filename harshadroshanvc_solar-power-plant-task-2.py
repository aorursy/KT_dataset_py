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
plant1_gd = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")
plant1_gd.info()
plant1_wsd = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
plant1_wsd.info()
plant1_gd['DATE_TIME'] = pd.to_datetime(plant1_gd['DATE_TIME'],format = '%d-%m-%Y %H:%M')
plant1_wsd['DATE_TIME'] = pd.to_datetime(plant1_wsd['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

plant1_gd['DATE'] = plant1_gd["DATE_TIME"].dt.date
plant1_gd['TIME'] = plant1_gd["DATE_TIME"].dt.time

plant1_wsd['DATE'] = plant1_wsd["DATE_TIME"].dt.date
plant1_wsd['TIME'] = plant1_wsd["DATE_TIME"].dt.time
fig,ax = plt.subplots(ncols=1,nrows=2,dpi=100,figsize=(20,10))
fig.tight_layout(h_pad = 6)

plant1_gd.plot(x = "DATE_TIME", y = "DC_POWER",ax = ax[0])
plant1_gd.plot(x = "DATE_TIME", y = "AC_POWER",ax = ax[1])

ax[0].set_title("DC_POWER over time")
ax[1].set_title("AC_POWER over time")
ax[0].set_ylabel("DC Power")
ax[1].set_ylabel("AC Power")

ax[0].tick_params(labelrotation=0)
ax[1].tick_params(labelrotation=0)
plant1_gd.set_index('TIME').drop('DATE_TIME',1)[["DC_POWER", "AC_POWER"]].plot(figsize = (20,5), style = 'o')
plt.title("Power over time interval")
plt.ylabel("POWER")
plt.xlabel("Time interval")
plant1_gd.groupby('DATE')[["DC_POWER", "AC_POWER"]].sum().plot(style = '--o', figsize = (20,5))
plt.xticks(rotation = '90')
plt.title("Total DC_POWER generated in each day")
plt.ylabel('Power')


plant1_gd.plot(x = "DATE_TIME", y = "DAILY_YIELD", figsize = (20,5), style = '--o')
plt.title("Daily yield over time")
plt.ylabel("Daily yield")
(plant1_gd.groupby('SOURCE_KEY')['TOTAL_YIELD'].max()).plot(kind = 'bar', figsize = (15,5))
plt.ylabel('Yield')
plt.xlabel('Generator')
plt.title('Total yield by each generator')
yield_daily = plant1_gd.groupby('DATE').sum()

fig,ax = plt.subplots(ncols=1,nrows=2,dpi=100,figsize=(20,10))
fig.tight_layout(h_pad = 6)

yield_daily.plot( y = "DAILY_YIELD",ax = ax[0], style = '--o')
yield_daily.plot(kind = 'bar', y = "TOTAL_YIELD",ax = ax[1])

ax[0].set_title("DAILY YEILD")
ax[1].set_title("TOTAL YIELD")
ax[0].set_ylabel("YIELD")
ax[1].set_ylabel("YIELD")

ax[0].tick_params(labelrotation=0)

plant1_gd.plot(x = "TIME", y = "DAILY_YIELD", figsize = (20,5), style = '--o')
plt.title("Daily yield over time interval")
plt.ylabel("Daily yield")
plant1_wsd.plot(x = "DATE_TIME", y = "IRRADIATION", figsize = (20,5), style = '--o')
plt.title("Irradiation over time")
plt.ylabel("Irradiation")
plant1_wsd.plot(x = "TIME", y = "IRRADIATION", figsize = (20,5), style = '--o')
plt.title("Irradiation over time interval")
plt.ylabel("Irradiation")
plant1_wsd.set_index('DATE_TIME')[['MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']].plot(figsize = (20,5))
plt.title("Module and Ambient temperature over time")
plt.ylabel("Temperature")
plant1_wsd.set_index('TIME')[['MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE']].plot(figsize = (20,5), style= '--o')
plt.title("Module and Ambient temperature over time")
plt.ylabel("Temperature")
