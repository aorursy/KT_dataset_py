# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.express as px

import plotly.offline as py

from plotly.subplots import make_subplots

import pycountry

py.init_notebook_mode(connected=True)

import folium 

from folium import plugins



# Graphics in retina format 

%config InlineBackend.figure_format = 'retina' 



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data['Last Update'] = pd.to_datetime(data['Last Update'])

data['Day'] = data['Last Update'].apply(lambda x : x.day)

data['Hour'] = data['Last Update'].apply(lambda x : x.hour)



data.drop(['Sno'], axis=1 , inplace=True)

data = data[data['Confirmed'] != 0]

data
df = pd.DataFrame(data.groupby('Day')['Confirmed'].sum())

df.reset_index(inplace=True)

fig = px.bar(df, y='Confirmed', x='Day', text='Confirmed')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig.show()
fig = make_subplots(rows=1, cols=3, specs=[[{"type" : "pie"}, {"type" : "pie"},{"type" : "pie"}]],

                    subplot_titles=("number of provience in countries", "Deaths", "Recovers"))



fig.add_trace(

    go.Pie(labels=data.groupby('Country')['Province/State'].nunique().sort_values(ascending=False)[:10].index,

           values=data.groupby('Country')['Province/State'].nunique().sort_values(ascending=False)[:10].values),

    row=1, col=1

)



fig.add_trace(

    go.Pie(labels=data[data.Deaths > 0].groupby('Country')["Deaths"].sum().index,

           values=data[data.Deaths > 0].groupby('Country')["Deaths"].sum().values),

    row=1, col=2

)

fig.add_trace(

    go.Pie(labels=data.groupby('Country')["Recovered"].sum().sort_values(ascending=False).index[:4],

           values=data.groupby('Country')["Recovered"].sum().sort_values(ascending=False).values[:4]),

    row=1, col=3

)



fig.update_layout(height=400, showlegend=True)

fig.show()
import plotly.graph_objects as go



days = list(data.Day.unique())



fig = go.Figure()

fig.add_trace(go.Bar(

    x=days,

    y=list(data.groupby('Day')['Confirmed'].sum()),

    name='Confirmed per day',

    marker_color='yellow'

))

fig.add_trace(go.Bar(

    x=days,

    y=list(data.groupby('Day')['Deaths'].sum()),

    name='Deaths per day',

    marker_color='red'

))



fig.add_trace(go.Bar(

    x=days,

    y=list(data.groupby('Day')['Recovered'].sum()),

    name='Recovered per day',

    marker_color='green'

))



# Here we modify the tickangle of the xaxis, resulting in rotated labels.

fig.update_layout(barmode='group', xaxis_tickangle=-45)

fig.show()


confirmed_countries = pd.DataFrame(data.groupby('Country')['Confirmed'].sum())

confirmed_countries.reset_index(inplace=True)



# Make a data frame with dots to show on the map

world_data = pd.DataFrame({

   'name':list(confirmed_countries['Country']),

    'lat':[-25.27,12.57,56.13, 35.8617,61.92,46.23,51.17,22.32,20.59,41.87,36.2,22.2,35.86,4.21,28.39,12.87,1.35,35.91,7.87,23.7,15.87,37.09,23.42,14.06,],

   'lon':[133.78,104.99,-106.35, 104.1954,25.75,2.21,10.45,114.17,78.96,12.56,138.25,113.54,104.19,101.98,84.12,121.77,103.82,127.77,80.77,120.96,100.99,-95.71,53.84,108.28],

   'Confirmed':list(confirmed_countries['Confirmed']),

})



# create map and display itconf

world_map = folium.Map(location=[10, -20], zoom_start=2.3,tiles='Stamen Toner')



for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Confirmed'], world_data['name']):

    folium.CircleMarker([lat, lon],

                        radius=10,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases from 22nd of jan</strong>: ' + str(value) + '<br>'),

                        color='red',

                        

                        fill_color='yellow',

                        fill_opacity=0.3).add_to(world_map)

world_map

#world_map.save('countries_affected.html')