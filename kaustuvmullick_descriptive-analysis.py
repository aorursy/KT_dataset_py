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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

df1_generation_data=pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

df1_sensor_data=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

df2_generation_data=pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

df2_sensor_data=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

                    
df1_generation_data.head

df1_sensor_data.head
df1_generation_data.shape

df1_sensor_data.shape
df1_generation_data.info()

df1_sensor_data.info()
df2_generation_data.head

df2_sensor_data.head
df2_generation_data.shape

df2_sensor_data.shape

df2_generation_data.info()

df2_sensor_data.info()
df1_generation_data.describe()

df1_sensor_data.describe()
df2_generation_data.describe()
df2_sensor_data.describe()
df1_generation_data.mean()

df1_sensor_data.mean()
df2_generation_data.mean()
df2_sensor_data.mean()
df1_sensor_data=df1_sensor_data.drop(['PLANT_ID','SOURCE_KEY'],axis=1)

df1_sensor_data.max()
df2_sensor_data=df2_sensor_data.drop(['PLANT_ID','SOURCE_KEY'],axis=1)

df2_sensor_data.max()
no_of_inverters_in_plant1=df1_generation_data['SOURCE_KEY'].nunique()

no_of_inverters_in_plant1
no_of_inverters_in_plant2=df2_generation_data['SOURCE_KEY'].nunique()

no_of_inverters_in_plant2
maximum_ac_power_plant1=df1_generation_data['AC_POWER'].max()

maximum_ac_power_plant1
maximum_dc_power_plant1=df1_generation_data['DC_POWER'].max()

maximum_dc_power_plant1
maximum_ac_power_plant2=df2_generation_data['AC_POWER'].max()

maximum_ac_power_plant2
maximum_dc_power_plant2=df2_generation_data['DC_POWER'].max()

maximum_dc_power_plant2
maximum_dc_power_inv_plant1=df1_generation_data.iloc[[df1_generation_data['DC_POWER'].max()]]

maximum_dc_power_inv_plant1
maximum_ac_power_inv_plant1=df1_generation_data.iloc[[df1_generation_data['AC_POWER'].max()]]

maximum_ac_power_inv_plant1
maximum_dc_power_inv_plant2=df2_generation_data.iloc[[df2_generation_data['DC_POWER'].max()]]

maximum_dc_power_inv_plant2
maximum_ac_power_inv_plant2=df2_generation_data.iloc[[df1_generation_data['AC_POWER'].max()]]

maximum_ac_power_inv_plant2
df1_generation_data=df1_generation_data.sort_values(['DC_POWER'],axis=0)

df1_generation_data
df1_generation_data=df1_generation_data.sort_values(['AC_POWER'],axis=0)

df1_generation_data
df2_generation_data=df2_generation_data.sort_values(['DC_POWER'],axis=0)

df2_generation_data
df2_generation_data=df2_generation_data.sort_values(['DC_POWER'],axis=0)

df2_generation_data
df1_generation_data.columns[df1_generation_data.isnull().any()]
df1_sensor_data.columns[df1_sensor_data.isnull().any()]
df2_generation_data.columns[df2_generation_data.isnull().any()]
df2_sensor_data.columns[df2_sensor_data.isnull().any()]