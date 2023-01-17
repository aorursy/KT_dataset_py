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
import pandas as pd
dgd1=pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

dws1=pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

dws1["DATE_TIME"]=pd.to_datetime(dws1["DATE_TIME"], format='%Y-%m-%d %H:%M:%S')  
dws1['DATE'] = pd.to_datetime(dws1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
dws1['TIME'] = pd.to_datetime(dws1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
dws1

dgd1["DATE_TIME"]=pd.to_datetime(dgd1["DATE_TIME"], format='%d-%m-%Y %H:%M')
dgd1['DATE'] = pd.to_datetime(dgd1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
dgd1['TIME'] = pd.to_datetime(dgd1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
dgd1


m_left = pd.merge(dgd1,dws1, on='DATE_TIME',how='left')
m_left

import matplotlib.pyplot as plt
#GRAPH OFAC_POWER and DC_POWER over 34 Days
plt.figure(figsize=(12,8))
plt.plot(dgd1["DATE_TIME"],dgd1['DC_POWER'].rolling(window=20).mean(),label="DC_POWER",c="yellow")
plt.plot(dgd1["DATE_TIME"],dgd1['AC_POWER'].rolling(window=20).mean(),label="AC_POWER")
plt.title('AC_POWER and DC_POWER over 34 Days ')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.legend()
plt.margins(0.05)
plt.show()
#GRAPH OF Ambient Temperature and Module Temperature over 34 Days 
plt.figure(figsize=(12,8))
plt.plot(dws1["DATE_TIME"],dws1['AMBIENT_TEMPERATURE'].rolling(window=20).mean(),label="AMBIENT",c="cyan")
plt.plot(dws1["DATE_TIME"],dws1['MODULE_TEMPERATURE'].rolling(window=20).mean(),label="MODULE",c="r")
plt.title('Ambient Temperature and Module Temperature over 34 Days ')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.legend()
plt.margins(0.05)
plt.show()
#GRAPH OF AC_POWER and DC_POWER over 34 Days
plt.figure(figsize=(12,8))
plt.plot(dgd1["DATE_TIME"],dgd1['DC_POWER'].rolling(window=20).mean(),label="DC_POWER",c="yellow")
plt.plot(dgd1["DATE_TIME"],dgd1['AC_POWER'].rolling(window=20).mean(),label="AC_POWER")
plt.plot(dgd1["DATE_TIME"],dgd1['DC_POWER']-dgd1["AC_POWER"].rolling(window=10).mean(),label="DIFFERENCE",c="pink")
plt.title('AC_POWER and DC_POWER over 34 Days ')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.legend()
plt.margins(0.05)
plt.show()

#GRAPH OFAmbient Temperature and Module Temperature over 34 Days
plt.figure(figsize=(12,8))
plt.plot(dws1["DATE_TIME"],dws1['AMBIENT_TEMPERATURE'].rolling(window=20).mean(),label="AMBIENT",c="cyan")
plt.plot(dws1["DATE_TIME"],dws1['MODULE_TEMPERATURE'].rolling(window=20).mean(),label="MODULE",c="r")
plt.plot(dws1["DATE_TIME"],dws1['MODULE_TEMPERATURE']-dws1["AMBIENT_TEMPERATURE"].rolling(window=10).mean(),label="DIFFERENCE",c="yellow")
plt.title('Ambient Temperature and Module Temperature over 34 Days ')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.legend()
plt.margins(0.05)
plt.show()
plt.figure(figsize=(12,8))
plt.plot(dws1["DATE_TIME"],dws1['IRRADIATION'].rolling(window=20).mean(),c="cyan")
plt.grid()
plt.margins(0.05)
plt.legend()
plt.title('irradiation over 34 days for plant 1')
plt.xlabel('Date and Time')
plt.ylabel('irradiation')
plt.show()
#GRAPH OF daily yield over 34 days for plant 1
plt.figure(figsize=(15,8))
plt.plot(dgd1["DATE_TIME"],dgd1['DAILY_YIELD'].rolling(window=20).mean(),c="purple")
plt.grid()
plt.margins(0.05)
plt.legend()
plt.title('daily yield over 34 days for plant 1')
plt.xlabel('Date and Time')
plt.ylabel('Daily yield')
plt.show()
#plant2
dgd2=pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")
dws2=pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
dgd2["DATE_TIME"]=pd.to_datetime(dgd2["DATE_TIME"], format='%Y-%m-%d %H:%M:%S')  
dgd2['DATE'] = pd.to_datetime(dgd2['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
dgd2['TIME'] = pd.to_datetime(dgd2['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
dgd2
dws2["DATE_TIME"]=pd.to_datetime(dws2["DATE_TIME"], format='%Y-%m-%d %H:%M:%S')  
dws2['DATE'] = pd.to_datetime(dws2['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
dws2['TIME'] = pd.to_datetime(dws2['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
dws2
dws2["DATE_TIME"]=pd.to_datetime(dws2["DATE_TIME"], format='%d-%m-%Y %H:%M')
dws2['DATE'] = pd.to_datetime(dws2['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date
dws2['TIME'] = pd.to_datetime(dws2['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.time
dws2

import seaborn as sns
#graph of date vs. dc power for plant2
sns.set_style('whitegrid');
sns.FacetGrid(dgd2,hue='SOURCE_KEY',height=12)\
   .map(plt.scatter,'DATE','DC_POWER')\
   .add_legend();
plt.show();
#graph of irradiation vs. datetime for plant2
sns.set_style('whitegrid');
sns.FacetGrid(dws2,hue='TIME',height=12)\
   .map(plt.scatter,'DATE_TIME','IRRADIATION')\
   .add_legend();
plt.show();
sns.set_style('whitegrid');
sns.FacetGrid(dws2,hue='DATE',height=15)\
   .map(plt.scatter,'DATE_TIME','MODULE_TEMPERATURE')\
   .add_legend();
plt.show();
sns.set_style('whitegrid');
sns.FacetGrid(dws2,hue='DATE',height=15)\
   .map(plt.scatter,'DATE_TIME','AMBIENT_TEMPERATURE')\
   .add_legend();
plt.show();
plt.figure(figsize=(12,8))
plt.plot(dws2["DATE_TIME"],dws2['AMBIENT_TEMPERATURE'].rolling(window=20).mean(),label="AMBIENT",c="red")
plt.plot(dws2["DATE_TIME"],dws2['MODULE_TEMPERATURE'].rolling(window=20).mean(),label="MODULE",c="purple")
plt.plot(dws2["DATE_TIME"],dws1['MODULE_TEMPERATURE']-dws2["AMBIENT_TEMPERATURE"].rolling(window=10).mean(),label="DIFFERENCE",c="yellow")
plt.margins(0.05)
plt.legend()
plt.title('Ambient and module temperature over 34 Days for plant2')
plt.xlabel('Date and Time')
plt.ylabel('temperature')
plt.show()
plt.figure(figsize=(15,9))
plt.plot(dgd2["DATE_TIME"],dgd2['DAILY_YIELD'].rolling(window=90).mean(),c="pink")
plt.grid()
plt.margins(0.05)
plt.legend()
plt.title('daily yield over 34 days for plant 2')
plt.xlabel('Date and Time')
plt.ylabel('Daily yield')
plt.show()
#graph of daily yield vs. date for plant2
sns.set_style('whitegrid');
sns.FacetGrid(dgd2,hue='SOURCE_KEY',height=12)\
   .map(plt.scatter,'DATE_TIME','DAILY_YIELD')\
   .add_legend();
plt.show();
#graph of ambient vs. module temperature for plant1 bydate
sns.set_style('whitegrid');
sns.FacetGrid(dws1,hue='DATE',size=12)\
   .map(plt.scatter,'MODULE_TEMPERATURE','AMBIENT_TEMPERATURE')\
   .add_legend();
plt.show();
#graph of ambient vs. module temperature for plant1

plt.figure(figsize=(15,8))
plt.plot(dws1.AMBIENT_TEMPERATURE,
        dws1.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.3,
        ms=10,
         c="pink",
        label='Module Temperature')
plt.grid()
plt.margins(0.05)
plt.legend()
plt.title('Ambient Temperature vs. Module Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()

#according todates
dates = dws1['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(20, 9))

for date in dates:
    dfws1 = dws1[dws1['DATE']==date]
    ax.plot(dfws1.AMBIENT_TEMPERATURE,
            dfws1.MODULE_TEMPERATURE,
            marker='o',
            linestyle='',
            alpha=.6,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs. Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#graph of dcpower vs acpower for plant1
dates = dgd1['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    dfgd1 = dgd1[dgd1['DATE']==date]

    ax.plot(dfgd1.AC_POWER,
            dfgd1.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date())

ax.margins(0.05)
ax.legend()
plt.title('Ac power vs. Dc power')
plt.xlabel('Dc power')
plt.ylabel('Ac power')
plt.show()
#for dc and ac power of plant1 according to sourcekey
sns.set_style('whitegrid');
sns.FacetGrid(dgd1,hue='SOURCE_KEY',size=12)\
   .map(plt.scatter,'DC_POWER','AC_POWER')\
   .add_legend();
plt.show();
#graph of irradiation vs. ambient temperature
sns.set_style('whitegrid');
sns.FacetGrid(dws1,hue='DATE',size=11)\
   .map(plt.scatter,'IRRADIATION','AMBIENT_TEMPERATURE')\
   .add_legend();
plt.show();
#graph of irradiation vs. module temperature
sns.set_style('whitegrid');
sns.FacetGrid(dws1,hue='DATE',size=11)\
   .map(plt.scatter,'IRRADIATION','MODULE_TEMPERATURE')\
   .add_legend();
plt.show();
plt.figure(figsize=(15,8))
plt.plot(dws1.IRRADIATION,
        dws1.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.3,
        ms=10,
         c="orange",
        label='Module Temperature')
plt.grid()
plt.margins(0.05)
plt.legend()
plt.title('Irradiation vs. Module Temperature')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#graph of ambient vs. module temperature for plant2 bydate
sns.set_style('whitegrid');
sns.FacetGrid(dws2,hue='DATE',size=12)\
   .map(plt.scatter,'MODULE_TEMPERATURE','AMBIENT_TEMPERATURE')\
   .add_legend();
plt.show();
#graph of ambient vs. module temperature for plant1

plt.figure(figsize=(15,8))
plt.plot(dws2.AMBIENT_TEMPERATURE,
        dws2.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.3,
        ms=10,
         c="red")
plt.grid()
plt.margins(0.05)
plt.legend()
plt.title('Ambient Temperature vs. Module Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()

#according todates
dates = dws2['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(20, 9))

for date in dates:
    dfws2 = dws2[dws2['DATE']==date]
    ax.plot(dfws2.AMBIENT_TEMPERATURE,
            dfws2.MODULE_TEMPERATURE,
            marker='o',
            linestyle='',
            alpha=.6,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date()
           )

ax.grid()
ax.margins(0.05)
ax.legend()
plt.title('Module Temperature vs. Ambient Temperature')
plt.xlabel('Ambient Temperature')
plt.ylabel('Module Temperature')
plt.show()
#for dc and ac power of plant2 according to sourcekey
sns.set_style('whitegrid');
sns.FacetGrid(dgd2,hue='SOURCE_KEY',size=12)\
   .map(plt.scatter,'DC_POWER','AC_POWER')\
   .add_legend();
plt.show();
plt.figure(figsize=(12,8))
plt.plot(dgd2["DATE_TIME"],dgd2['DC_POWER'].rolling(window=20).mean(),label="DC_POWER",c="red")
plt.plot(dgd2["DATE_TIME"],dgd2['AC_POWER'].rolling(window=20).mean(),label="AC_POWER")
plt.title('AC_POWER and DC_POWER over 34 Days ')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.legend()
plt.margins(0.05)
plt.show()
plt.figure(figsize=(12,8))
plt.plot(dgd2["DATE_TIME"],dgd2['DC_POWER'].rolling(window=20).mean(),label="DC_POWER",c="yellow")
plt.plot(dgd2["DATE_TIME"],dgd2['AC_POWER'].rolling(window=20).mean(),label="AC_POWER")
plt.plot(dgd2["DATE_TIME"],dgd2['DC_POWER']-dgd2["AC_POWER"].rolling(window=60).mean(),label="DIFFERENCE",c="purple")
plt.title('AC_POWER and DC_POWER over 34 Days ')
plt.xlabel('Date and Time')
plt.ylabel('Temperature')
plt.legend()
plt.margins(0.05)
plt.show()
#graph of irradiation vs. ambient temperature for plant2
sns.set_style('whitegrid');
sns.FacetGrid(dws2,hue='DATE',size=11)\
   .map(plt.scatter,'IRRADIATION','AMBIENT_TEMPERATURE')\
   .add_legend();
plt.show();
#Irradiation vs. Module Temperature forplant 2
plt.figure(figsize=(15,8))
plt.plot(dws2.IRRADIATION,
        dws2.MODULE_TEMPERATURE,
        marker='o',
        linestyle='',
        alpha=.3,
        ms=10,
         c="purple",
        label='Module Temperature')
plt.grid()
plt.margins(0.05)
plt.legend()
plt.title('Irradiation vs. Module Temperature')
plt.xlabel('Irradiation')
plt.ylabel('Module Temperature')
plt.show()
#graph of dcpower vs acpower for plant2
dates = dgd2['DATE'].unique()

_, ax = plt.subplots(1, 1, figsize=(18, 9))

for date in dates:
    dfgd2 = dgd2[dgd2['DATE']==date]

    ax.plot(dfgd2.AC_POWER,
            dfgd2.DC_POWER,
            marker='o',
            linestyle='',
            alpha=.5,
            ms=10,
            label=pd.to_datetime(date,format='%Y-%m-%d').date())

ax.margins(0.05)
ax.legend()
plt.title('Ac power vs. Dc power')
plt.xlabel('Dc power')
plt.ylabel('Ac power')
plt.show()
#graph of irradiation vs. dc power 
sns.set_style('whitegrid');
sns.FacetGrid(m_left,hue='DATE_y',size=12)\
   .map(plt.scatter,'IRRADIATION','DC_POWER')\
   .add_legend();
plt.show();
plt.figure(figsize=(15,8))
plt.plot(m_left["IRRADIATION"],
       m_left["AC_POWER"],c="cyan",alpha=0.3,marker="o",linestyle="")
plt.xlabel("Irradiation")
plt.ylabel("AC power")
plt.margins(0.05)
plt.legend()
plt.title('Irradiation vs. AC_POWER')
plt.grid()
plt.show()
