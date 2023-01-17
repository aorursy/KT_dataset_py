import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import seaborn as sns

import plotly.graph_objs as go

import plotly.plotly as py

sns.set(style="darkgrid")

import matplotlib.pyplot as plt

%matplotlib inline

import os
path2 = r'../input/input_data/weather' # use your path

all_files2 = glob.glob(path2 + "/*.parquet")



li2 = []



for filename in all_files2:

    df2 = pd.read_parquet(filename, engine='auto')

    li2.append(df2)



weatherframe = pd.concat(li2, axis=0, ignore_index=True)
weatherframe.describe(include='all').T
weatherframe['weather_condition']=np.where(weatherframe['temperature_data']<=270.372,'Snow',

                                           np.where(weatherframe['temperature_data']<=273.15,'Freezing Rain','Rain'))

weatherframe['weather_Percp']=np.where(weatherframe['precipitation_data']!=0,

                                       np.where((weatherframe['precipitation_data']<=2.5),'Low',

                                       np.where(weatherframe['precipitation_data']<=7.6,'Moderate','High')),'No Rain')
weatherframe.temperature_data=weatherframe.temperature_data.astype(int)

weatherframe.precipitation_data=weatherframe.precipitation_data.astype(int)
plt.figure(figsize=(13, 4))

sns.distplot(weatherframe['temperature_data'])
# Precipitation data excluding 0 (since most of them are 0)

plt.figure(figsize=(13, 4))

sns.distplot(weatherframe[weatherframe['precipitation_data']>0].precipitation_data)
weather_conditions=weatherframe.groupby('weather_condition')['date'].count().reset_index()

weather_conditions=weather_conditions.rename(columns={'date':'number'}).sort_values(by='number')

sns.barplot(y=weather_conditions.weather_condition, x=weather_conditions.number,orient='h', data=weather_conditions,color='darkcyan',order=weather_conditions.weather_condition)

precipitation_conditions=weatherframe.groupby('weather_Percp')['date'].count().reset_index()

precipitation_conditions=precipitation_conditions.rename(columns={'date':'number'}).sort_values(by='number')

sns.barplot(y=precipitation_conditions.weather_Percp, x=precipitation_conditions.number,orient='h', data=precipitation_conditions,color='darkcyan',order=precipitation_conditions.weather_Percp)

import IPython

IPython.display.IFrame("https://www.google.com/maps/d/u/0/embed?mid=1_Y6YmiirI2JckTwaim6rWHVHTsCsm5SZ" ,width="900", height="350")