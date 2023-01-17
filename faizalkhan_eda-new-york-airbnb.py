# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head(6)
data.shape
total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum())*100)/data.isnull().count().sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'], sort=False).sort_values('Total', ascending=False)

missing_data.head(40)
data.describe().T
import matplotlib.pyplot as plt

import seaborn as sns 

plt.figure(figsize=(10,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group)

plt.ioff()
import plotly.offline as pyo

import plotly.graph_objs as go

roomdf = data.groupby('room_type').size()/data['room_type'].count()*100

labels = roomdf.index

values = roomdf.values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.7)])

fig.show()
import folium

from folium.plugins import HeatMap

m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(data[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
plt.figure(figsize=(10,6))

sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = data)

plt.title("Room types occupied by the neighbourhood_group")

plt.show()
plt.figure(figsize=(10,6))

data['number_of_reviews'].plot(kind='hist')

plt.xlabel("Price")

plt.ioff()

plt.show()
df1=data.sort_values(by=['room_type'],ascending=False).head(1000)

import folium

from folium.plugins import MarkerCluster

from folium import plugins

print('sample of 1000 rooms')

Long=-73.80

Lat=40.80

mapdf1=folium.Map([Lat,Long],zoom_start=10,)



mapdf1_rooms_map=plugins.MarkerCluster().add_to(mapdf1)



for lat,lon,label in zip(df1.latitude,df1.longitude,df1.name):

    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)

mapdf1.add_child(mapdf1_rooms_map)



mapdf1