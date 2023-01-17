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
#All necessary imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Initiate Data Frames

plant1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

plant1_w = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

plant2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

plant2_w = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
#Convert DATE_TIME type from str to datetime

plant1['DATE_TIME'] = pd.to_datetime(plant1['DATE_TIME'])

plant2['DATE_TIME'] = pd.to_datetime(plant2['DATE_TIME'])

plant1_w['DATE_TIME'] = pd.to_datetime(plant1_w['DATE_TIME'])

plant2_w['DATE_TIME'] = pd.to_datetime(plant2_w['DATE_TIME'])
#sort by datetime

plant1 = plant1.sort_values(by='DATE_TIME')

plant2 = plant2.sort_values(by='DATE_TIME')

plant1_w = plant1_w.sort_values(by='DATE_TIME')

plant2_w = plant2_w.sort_values(by='DATE_TIME')
#PLANT-1 DC POWER VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='DC_POWER',data=plant1)

plt.title('PLANT-1 DC POWER VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('DC POWER',size='x-large')
#PLANT-1 AC POWER VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='AC_POWER',data=plant1)

plt.title('PLANT-1 AC POWER VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('AC POWER',size='x-large')
#PLANT-2 DC POWER VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='DC_POWER',data=plant2)

plt.title('PLANT-2 DC POWER VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('DC POWER',size='x-large')
#PLANT-2 AC POWER VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='AC_POWER',data=plant2)

plt.title('PLANT-2 AC POWER VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('AC POWER',size='x-large')
#PLANT-1 IRRADIATION VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='IRRADIATION',data=plant1_w)

plt.title('PLANT-1 IRRADIATION VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('IRRADIATION',size='x-large')
#PLANT-2 IRRADIATION VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='IRRADIATION',data=plant2_w)

plt.title('PLANT-2 IRRADIATION VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('IRRADIATION',size='x-large')
#PLANT-1 AMBIENT TEMPERATURE VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='AMBIENT_TEMPERATURE',data=plant1_w)

plt.title('PLANT-1 AMBIENT TEMPERATURE VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('AMBIENT TEMPERATURE',size='x-large')

plt.tight_layout()
#PLANT-2 AMBIENT TEMPERATURE VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='AMBIENT_TEMPERATURE',data=plant2_w)

plt.title('PLANT-2 AMBIENT TEMPERATURE VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('AMBIENT TEMPERATURE',size='x-large')

plt.tight_layout()
#PLANT-1 MODULE TEMPERATURE VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='MODULE_TEMPERATURE',data=plant1_w)

plt.title('PLANT-1 MODULE TEMPERATURE VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('MODULE TEMPERATURE',size='x-large')

plt.tight_layout()
#PLANT-2 MODULE TEMPERATURE VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='MODULE_TEMPERATURE',data=plant2_w)

plt.title('PLANT-2 MODULE TEMPERATURE VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('MODULE TEMPERATURE',size='x-large')

plt.tight_layout()
#PLANT-1 DAILY YIELD VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='DAILY_YIELD',data=plant1)

plt.title('PLANT-1 DAILY YIELD VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('DAILY YIELD',size='x-large')

plt.tight_layout()
#PLANT-2 DAILY YIELD VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='DAILY_YIELD',data=plant2)

plt.title('PLANT-2 DAILY YIELD VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('DAILY YIELD',size='x-large')

plt.tight_layout()
#PLANT-1 TOTAL YIELD VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='TOTAL_YIELD',data=plant1)

plt.title('PLANT-1 TOTAL YIELD VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('TOTAL YIELD',size='x-large')

plt.tight_layout()
#PLANT-2 TOTAL YIELD VARIATION

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DATE_TIME',y='TOTAL_YIELD',data=plant2)

plt.title('PLANT-2 TOTAL YIELD VARIATION',size='xx-large')

plt.xlabel('DATE',size='x-large')

plt.ylabel('TOTAL YIELD',size='x-large')

plt.tight_layout()
#PLANT-1 DC Power VS AC Power

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DC_POWER',y='AC_POWER',data=plant1)

plt.title('PLANT-1 DC Power VS AC Power',size='xx-large')

plt.xlabel('DC POWER',size='x-large')

plt.ylabel('AC POWER',size='x-large')

plt.tight_layout()
#PLANT-2 DC Power VS AC Power

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='DC_POWER',y='AC_POWER',data=plant2)

plt.title('PLANT-2 DC Power VS AC Power',size='xx-large')

plt.xlabel('DC POWER',size='x-large')

plt.ylabel('AC POWER',size='x-large')

plt.tight_layout()
#PLANT-1 AMBIENT VS MODULE TEMPERATURE

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='AMBIENT_TEMPERATURE',y='MODULE_TEMPERATURE',data=plant1_w)

plt.title('PLANT-1 AMBIENT VS MODULE TEMPERATURE',size='xx-large')

plt.xlabel('AMBIENT TEMPERATURE',size='x-large')

plt.ylabel('MODULE TEMPERATURE',size='x-large')

plt.tight_layout()
#PLANT-2 AMBIENT VS MODULE TEMPERATURE

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='AMBIENT_TEMPERATURE',y='MODULE_TEMPERATURE',data=plant2_w)

plt.title('PLANT-2 AMBIENT VS MODULE TEMPERATURE',size='xx-large')

plt.xlabel('AMBIENT TEMPERATURE',size='x-large')

plt.ylabel('MODULE TEMPERATURE',size='x-large')

plt.tight_layout()
#PLANT-1 IRRADIATION VS AMBIENT TEMPERATURE

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='IRRADIATION',y='AMBIENT_TEMPERATURE',data=plant1_w)

plt.title('PLANT-1 IRRADIATION VS AMBIENT TEMPERATURE',size='xx-large')

plt.xlabel('IRRADIATION',size='x-large')

plt.ylabel('AMBIENT TEMPERATURE',size='x-large')

plt.tight_layout()
#PLANT-2 IRRADIATION VS AMBIENT TEMPERATURE

plt.figure(figsize=[12,6])

sns.set_style('white')

sns.lineplot(x='IRRADIATION',y='AMBIENT_TEMPERATURE',data=plant2_w)

plt.title('PLANT-2 IRRADIATION VS AMBIENT TEMPERATURE',size='xx-large')

plt.xlabel('IRRADIATION',size='x-large')

plt.ylabel('AMBIENT TEMPERATURE',size='x-large')

plt.tight_layout()