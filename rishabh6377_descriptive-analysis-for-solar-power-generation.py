import warnings

warnings.filterwarnings('ignore')
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
plant_1_sensor = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

plant_2_sensor = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")

plant_1_generation = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")

plant_2_generation = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv")
print(plant_1_sensor.shape,plant_1_generation.shape,plant_2_sensor.shape,plant_2_generation.shape)
plant_1_sensor.head()
plant_1_generation.head()
plant_2_sensor.head()
plant_2_generation[plant_2_generation.DC_POWER > 1000]
print("*******************Plant_1_Sensor*****************")

print(plant_1_sensor.nunique()/plant_1_sensor.shape[0]*100)

print("*******************Plant_1_Generation*****************")

print(plant_1_generation.nunique()/plant_1_generation.shape[0]*100)

print("*******************Plant_2_Sensor*****************")

print(plant_2_sensor.nunique()/plant_2_sensor.shape[0]*100)

print("*******************Plant_2_Generation*****************")

print(plant_2_generation.nunique()/plant_2_generation.shape[0]*100)
print(plant_1_sensor.info())

print(plant_1_generation.info())

print(plant_2_sensor.info())

print(plant_2_generation.info())
from datetime import datetime
plant_1_generation['DATE'] = [datetime.strftime(i,format="%d-%m-%Y") for i in pd.to_datetime(plant_1_generation.DATE_TIME)]

plant_1_generation['DATE'] = pd.to_datetime(plant_1_generation['DATE'])
plant_2_generation['DATE'] = [datetime.strftime(i,format="%Y-%m-%d") for i in pd.to_datetime(plant_2_generation.DATE_TIME)]

plant_2_generation['DATE'] = pd.to_datetime(plant_2_generation['DATE'])
plant_1_sensor['DATE'] = [datetime.strftime(i,format="%Y-%m-%d") for i in pd.to_datetime(plant_1_sensor.DATE_TIME)]

plant_1_sensor['DATE'] = pd.to_datetime(plant_1_sensor['DATE'])
plant_2_sensor['DATE'] = [datetime.strftime(i,format="%Y-%m-%d") for i in pd.to_datetime(plant_2_sensor.DATE_TIME)]

plant_2_sensor['DATE'] = pd.to_datetime(plant_2_sensor['DATE'])
plant_1_generation['YEAR'] = [datetime.strftime(i,format="%Y") for i in plant_1_generation.DATE]

plant_1_generation['MONTH'] = [datetime.strftime(i,format="%m") for i in plant_1_generation.DATE]

plant_1_generation['DAY'] = [datetime.strftime(i,format="%d") for i in plant_1_generation.DATE]
plant_2_generation['YEAR'] = [datetime.strftime(i,format="%Y") for i in plant_2_generation.DATE]

plant_2_generation['MONTH'] = [datetime.strftime(i,format="%m") for i in plant_2_generation.DATE]

plant_2_generation['DAY'] = [datetime.strftime(i,format="%d") for i in plant_2_generation.DATE]
plant_1_sensor['YEAR'] = [datetime.strftime(i,format="%Y") for i in plant_1_sensor.DATE]

plant_1_sensor['MONTH'] = [datetime.strftime(i,format="%m") for i in plant_1_sensor.DATE]

plant_1_sensor['DAY'] = [datetime.strftime(i,format="%d") for i in plant_1_sensor.DATE]
plant_2_sensor['YEAR'] = [datetime.strftime(i,format="%Y") for i in plant_2_sensor.DATE]

plant_2_sensor['MONTH'] = [datetime.strftime(i,format="%m") for i in plant_2_sensor.DATE]

plant_2_sensor['DAY'] = [datetime.strftime(i,format="%d") for i in plant_2_sensor.DATE]
sns.set_style('whitegrid')

plt.figure(figsize=(14,8))

fig=sns.lineplot(data=plant_1_generation,x='DATE',y='DAILY_YIELD',ci=None)

fig.set_title('Plant 1 Daily Power Generation')

plt.show()
sns.set_style('whitegrid')

plt.figure(figsize=(16,8))

sns.lineplot(data=plant_2_generation,x='DATE',y='DAILY_YIELD',ci=None)

plt.title('Plant 2 Daily Power Generation')

plt.show()
plant_1_daily_yield_mean = plant_1_generation.groupby(['DATE']).DAILY_YIELD.sum().mean() 

plant_2_daily_yield_mean = plant_2_generation.groupby(['DATE']).DAILY_YIELD.sum().mean()
print(plant_1_daily_yield_mean,plant_2_daily_yield_mean)
sns.set_style('darkgrid')

plt.figure(figsize=(10,6))

plt.bar(x=['Plant_1','Plant_2'],height=[plant_1_daily_yield_mean,plant_2_daily_yield_mean],color=["green","blue"])

for i,j in enumerate([plant_1_daily_yield_mean,plant_2_daily_yield_mean]):

    plt.text(i-0.2,j+100000,str(j))

plt.title("mean value of daily yield")

plt.yticks(range(0,8000001,1000000))

plt.show()
plant_1_irradiation = plant_1_sensor.groupby(['DATE']).IRRADIATION.sum()

plant_2_irradiation = plant_2_sensor.groupby(['DATE']).IRRADIATION.sum()
pd.DataFrame({'Date':plant_1_irradiation.index,'Plant_1_Daily_Irradiation':plant_1_irradiation.values,'Plant_2_Daily_Irradiation':plant_2_irradiation.values})
max_amb_modu_temp =pd.DataFrame({'Max_Ambient':[plant_1_sensor.AMBIENT_TEMPERATURE.max(),plant_2_sensor.AMBIENT_TEMPERATURE.max()],

                                 'Max_Module':[plant_1_sensor.MODULE_TEMPERATURE.max(),plant_2_sensor.MODULE_TEMPERATURE.max()]},index=['Plant_1','Plant_2'])

max_amb_modu_temp
print("No. of inverters in Plant_1 : ",len(plant_1_generation.SOURCE_KEY.value_counts()))

print("No. of inverters in Plant_2 : ",len(plant_2_generation.SOURCE_KEY.value_counts()))
pd.DataFrame({'DC_Power_Max':plant_1_generation.groupby(['DATE']).DC_POWER.max().values,'DC_Power_Min':plant_1_generation.groupby(['DATE']).DC_POWER.min().values,

              'AC_Power_Max':plant_1_generation.groupby(['DATE']).AC_POWER.max().values,'AC_Power_Min':plant_1_generation.groupby(['DATE']).AC_POWER.min().values},

             index=plant_1_generation.groupby(['DATE']).DC_POWER.max().index)
pd.DataFrame({'DC_Power_Max':plant_2_generation.groupby(['DATE']).DC_POWER.max().values,'DC_Power_Min':plant_2_generation.groupby(['DATE']).DC_POWER.min().values,

              'AC_Power_Max':plant_2_generation.groupby(['DATE']).AC_POWER.max().values,'AC_Power_Min':plant_2_generation.groupby(['DATE']).AC_POWER.min().values},

             index=plant_2_generation.groupby(['DATE']).DC_POWER.max().index)
dc_ac =pd.DataFrame({'Inverter_Source_Key':[plant_1_generation.groupby(['SOURCE_KEY']).DC_POWER.sum().sort_values(ascending=False).index[0],

                    plant_2_generation.groupby(['SOURCE_KEY']).DC_POWER.sum().sort_values(ascending=False).index[0]],

                    'Max_DC':[plant_1_generation.groupby(['SOURCE_KEY']).DC_POWER.sum().max(),plant_2_generation.groupby(['SOURCE_KEY']).DC_POWER.sum().max()],

                        'Max_AC':[plant_1_generation.groupby(['SOURCE_KEY']).AC_POWER.sum().max(),plant_2_generation.groupby(['SOURCE_KEY']).AC_POWER.sum().max()],

                     },index=['Plant_1','Plant_2'])

dc_ac
index_dc_p1 = plant_1_generation.groupby(['SOURCE_KEY']).DC_POWER.sum().sort_values(ascending=False).index

index_ac_p1 = plant_1_generation.groupby(['SOURCE_KEY']).AC_POWER.sum().sort_values(ascending=False).index
#Reason to create this is so that we can create a single dataframe for both AC and DC production ranking.

#Through this we can give excat rating to each inverter w.r.t their production.



ac_dict_p1 = dict() 

for i,j in zip(index_dc_p1,range(1,23,1)):

    ac_dict_p1[i] = j

ac_pr_per_dc_rank = []

for i in index_dc_p1:

    ac_pr_per_dc_rank.append(ac_dict_p1[i])

ac_pr_per_dc_rank
plant_1_dc_ac_rank = pd.DataFrame({'Plant_1_Inverter_Source_Key':index_dc_p1,'DC_Production_Rank':range(1,23,1),'AC_Production_Rank':ac_pr_per_dc_rank})

plant_1_dc_ac_rank
index_dc_p2 = plant_2_generation.groupby(['SOURCE_KEY']).DC_POWER.sum().sort_values(ascending=False).index

index_ac_p2 = plant_2_generation.groupby(['SOURCE_KEY']).AC_POWER.sum().sort_values(ascending=False).index
ac_dict_p2 = dict()

for i,j in zip(index_dc_p2,range(1,23,1)):

    ac_dict_p2[i] = j

ac_pr_per_dc_rank_p2 = []

for i in index_dc_p2:

    ac_pr_per_dc_rank_p2.append(ac_dict_p2[i])

ac_pr_per_dc_rank_p2
plant_2_dc_ac_rank = pd.DataFrame({'Plant_2_Inverter_Source_Key':index_dc_p2,'DC_Production_Rank':range(1,23,1),'AC_Production_Rank':ac_pr_per_dc_rank_p2})

plant_2_dc_ac_rank
print("*******************Plant_1_Sensor*****************")

print(plant_1_sensor.isna().sum()/plant_1_sensor.shape[0]*100)

print("*******************Plant_1_Generation*****************")

print(plant_1_generation.isna().sum()/plant_1_generation.shape[0]*100)

print("*******************Plant_2_Sensor*****************")

print(plant_2_sensor.isna().sum()/plant_2_sensor.shape[0]*100)

print("*******************Plant_2_Generation*****************")

print(plant_2_generation.isna().sum()/plant_2_generation.shape[0]*100)