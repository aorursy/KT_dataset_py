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
summary = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

measurement_item = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')

measurement_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv')

measurement_station = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
summary.head()
df = summary
df['Date'] =  df['Measurement date'].str[:10]

df['Time'] = df['Measurement date'].str[11:13]
df.tail()
df['Month'] = df['Date'].str[5:7]

df['Year'] = df['Date'].str[:4]

df['Day'] = df['Date'].str[8:10]
df.head()
#We first look at how each metric for pollution are distributed

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt  

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
import seaborn as sns

numerical_list = ['SO2','SO2','O3','CO','PM10','PM2.5']

for i in numerical_list:

    sns.distplot(df[i], hist=False, rug=True)

    plt.show();
#We will proceed with 2 metrics, PM2.5 and PM10 and see how certain factors affect them!
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='Station code', y='PM2.5', data=df, showfliers=False);

#Pm2.5 Levels by Stations
#Station Code 119 and 121 are a bit higher compared to 112 or 116

#Lets see how far are they situated
temp = df[(df['Station code'] == 119) | (df['Station code'] == 112)]

temp = temp.groupby(['Latitude','Longitude','Station code'])['Station code'].count().reset_index(name='count')

temp
import plotly.express as px

fig = px.density_mapbox(temp, lat='Latitude', lon='Longitude', radius=10,

                        center=dict(lat=37.5, lon=126.8), zoom=8,

                        mapbox_style="stamen-terrain")

fig.show()
#As you might observe that the lower point ie station code 119 is more in the city center hence it would be expected to recorder higher levels of PM2.5

#So definetly location will be a key factor
#Lets not look at how is each month over the years performs as a function of pollutions in PM2.5
temp = df.groupby(['Month'])['PM2.5'].mean().reset_index(name='Average')

temp
fig = px.bar(temp, x='Month', y='Average',color='Month')

fig.show()
#The month of March observes on average maximum PM2.5 Levels
fig, axes = plt.subplots(1, 1, figsize=(20, 6))

sns.boxplot(x='Month', y='PM2.5', data=df, showfliers=False);

#Pm2.5 Levels by Stations
#Looking at the boxplots gives similar results. The variation from months 6-9 is small probably due to rains during those months.