# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plant_1_weather_sensor_data = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

plant_1_generation_data = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv')

plant_2_weather_sensor_data = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')

plant_2_generation_data = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_2_Generation_Data.csv')

print('success')
plant_1_weather_sensor_data.describe()
plant_1_generation_data.describe()
plant_2_weather_sensor_data.describe()
plant_2_generation_data.describe()
plant_1_weather_sensor_data.head()
plant_1_generation_data.head()
plant_2_weather_sensor_data.head()
plant_2_generation_data.head()
plant_1_weather_sensor_data.isnull().sum()
plant_1_generation_data.isnull().sum()
plant_1_weather_sensor_data.isnull().sum()
plant_2_generation_data.isnull().sum()
plant_1_irradiation = plant_1_weather_sensor_data['IRRADIATION']

sum(plant_1_irradiation)
plant_2_irradiation = plant_2_weather_sensor_data['IRRADIATION']

sum(plant_2_irradiation)
plant_1_ambient_temperature = plant_1_weather_sensor_data['AMBIENT_TEMPERATURE']

max(plant_1_ambient_temperature)
plant_2_ambient_temperature = plant_2_weather_sensor_data['AMBIENT_TEMPERATURE']

max(plant_2_ambient_temperature)
plant_1_weather_sensor_data.size
plant_2_weather_sensor_data.size
plant_2_generation_data.iloc[[max(plant_2_generation_data['DC_POWER'])]]
plant_1_generation_data.iloc[[max(plant_1_generation_data['DC_POWER'])]]
plant_2_generation_data.sort_values("DC_POWER", axis = 0, ascending = False, 

                 inplace = True, na_position ='last') 

plant_2_generation_data
plant_1_generation_data.sort_values("DC_POWER", axis = 0, ascending = False, 

                 inplace = True, na_position ='last') 

plant_1_generation_data
df_date = plant_1_generation_data['DATE_TIME'].str.split()

x = []

for d in df_date:

    va = d[0]

    x.append(va)

plant_1_generation_data['DATE'] = pd.DataFrame(x)

plant_1_generation_data.drop(['DATE_TIME'],axis=1,inplace=True)
plant_1_generation = plant_1_generation_data.groupby("DATE")['SOURCE_KEY'].count().reset_index()

df = plant_1_generation.sort_values('DATE',ascending=False)

plt.figure(figsize=(15,5))

chart = sns.barplot(

    data = plant_1_generation,

    x = 'DATE',

    y = 'SOURCE_KEY',

    palette = 'Set1'

)

chart = chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light',

)

plt.show()
plant_1_generation = plant_1_generation_data.groupby("DATE")['DC_POWER'].count().reset_index()

df = plant_1_generation.sort_values('DATE',ascending=False)

plt.figure(figsize=(15,5))

chart = sns.barplot(

    data = plant_1_generation,

    x = 'DATE',

    y = 'DC_POWER',

    palette = 'Set1'

)

chart = chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light',

)

plt.show()
plant_1_generation = plant_1_generation_data.groupby("DATE")['AC_POWER'].count().reset_index()

df = plant_1_generation.sort_values('DATE',ascending=False)

plt.figure(figsize=(15,5))

chart = sns.barplot(

    data = plant_1_generation,

    x = 'DATE',

    y = 'AC_POWER',

    palette = 'Set1'

)

chart = chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light',

)

plt.show()
df_date = plant_2_generation_data['DATE_TIME'].str.split()

x = []

for d in df_date:

    va = d[0]

    x.append(va)

plant_2_generation_data['DATE'] = pd.DataFrame(x)

plant_2_generation_data.drop(['DATE_TIME'],axis=1,inplace=True)
plant_2_generation = plant_2_generation_data.groupby("DATE")['SOURCE_KEY'].count().reset_index()

df = plant_2_generation.sort_values('DATE',ascending=False)

plt.figure(figsize=(15,5))

chart = sns.barplot(

    data = plant_2_generation,

    x = 'DATE',

    y = 'SOURCE_KEY',

    palette = 'Set2'

)

chart = chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light',

)

plt.show()
plant_2_generation = plant_2_generation_data.groupby("DATE")['DC_POWER'].count().reset_index()

df = plant_2_generation.sort_values('DATE',ascending=False)

plt.figure(figsize=(15,5))

chart = sns.barplot(

    data = plant_1_generation,

    x = 'DATE',

    y = 'AC_POWER',

    palette = 'Set2'

)

chart = chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light',

)

plt.show()
plant_2_generation = plant_2_generation_data.groupby("DATE")['AC_POWER'].count().reset_index()

df = plant_2_generation.sort_values('DATE',ascending=False)

plt.figure(figsize=(15,5))

chart = sns.barplot(

    data = plant_1_generation,

    x = 'DATE',

    y = 'AC_POWER',

    palette = 'Set2'

)

chart = chart.set_xticklabels(

    chart.get_xticklabels(), 

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light',

)

plt.show()