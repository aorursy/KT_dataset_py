import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import figure, show

from matplotlib.ticker import MaxNLocator

import os 

import time

from matplotlib import cm

import pylab as pl
data=pd.read_csv("../input/crimes-in-boston/crime.csv",encoding="gbk")

data=data.loc[(data['Lat']>35)&(data['Long']< -60)] #remove NA from 'Lat' and 'Long'

data=data.dropna(subset=["STREET"])

columns=['OFFENSE_CODE_GROUP']

for j in columns:

	print(j,data[j].unique())



print(data.isnull().sum())
type={label: idx for idx, label in enumerate(np.unique(data['OFFENSE_CODE_GROUP']))}

data['type']=data['OFFENSE_CODE_GROUP'].map(type) #change crime type to number

print(type)

index=pd.Index(data['type'])

#print(index.value_counts().sort_values())
count=data['OFFENSE_CODE_GROUP'].value_counts()

groups=list(data['OFFENSE_CODE_GROUP'].value_counts().index)[:9]

counts=list(count[:9])

counts.append(count.agg(sum)-count[:9].agg('sum'))



groups.append('other_type')

type_dict={"group":groups,

          "counts":counts}

type_dict=pd.DataFrame(type_dict)

qx = type_dict.plot(kind='pie', figsize=(10,7), y='counts', labels=groups,

             autopct='%1.1f%%', pctdistance=0.9, radius=1.2)

plt.legend(loc=0, bbox_to_anchor=(0.95,0.6)) 



plt.title('Top 10 for crime type', weight='bold', size=14,y=1.08)

plt.axis('equal')

plt.ylabel('')

plt.show()

plt.clf()

plt.close()
print(data['OFFENSE_CODE_GROUP'].value_counts())
plt.bar(index.value_counts().index,index.value_counts())

plt.xlabel("Crime type")

plt.ylabel("Count")

plt.title("Counting for Crime type")

plt.show()

Crime_year=pd.Index(data['YEAR'])

ax =figure().gca()

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.bar(Crime_year.value_counts().index,Crime_year.value_counts())

plt.xlabel("Year")

plt.ylabel("Count")

plt.title("Counting the number for Crime (Year)")

plt.show(ax)



data.groupby(['YEAR','MONTH'])['OFFENSE_CODE_GROUP'].agg('count').unstack('YEAR')

fig,ax=plt.subplots(figsize=(15,6))

data.groupby(['MONTH','YEAR'])['OFFENSE_CODE_GROUP'].agg('count').unstack().plot(ax=ax)

plt.title("Counting the number for Crime (month)")

plt.grid(True)



data.groupby(['YEAR','MONTH'])['OFFENSE_CODE_GROUP'].agg('count').unstack('YEAR')

fig,ax=plt.subplots(figsize=(18,12))

data.groupby(['MONTH','YEAR'])['OFFENSE_CODE_GROUP'].agg('count').unstack().plot(kind='bar',ax=ax)

pl.xticks(rotation=360)

plt.title("Counting the number for Crime (month)")

plt.grid(True)
import geopandas as gpd

import folium





incidents=folium.map.FeatureGroup()



#for lat,lon, in zip(data.Lat,data.Long):

#	incidents.add_child(folium.CircleMarker([lat,lon],radius=7,color='yellow',fill=True,fill_color='red',fill_opacity=0.4))



Lat=42.3

Lon=-71.1

#boston_map=folium.Map([Lat,Lon],zoom_start=12)

#boston_map.add_child(incidents)

#boston_map.save("mymap.html")



from folium import plugins



data1=data[data['YEAR']==2015][0:2000]

filename="Crime2015"

boston_map=folium.Map([Lat,Lon],zoom_start=12)

incidents2=plugins.MarkerCluster().add_to(boston_map)

for lat,lon,label in zip(data1.Lat,data1.Long,data1.OFFENSE_CODE_GROUP):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(incidents2)

boston_map.add_child(incidents2)

boston_map

data1=data[data['YEAR']==2016][0:2000]

filename="Crime2016"

boston_map=folium.Map([Lat,Lon],zoom_start=12)

incidents2=plugins.MarkerCluster().add_to(boston_map)

for lat,lon,label in zip(data1.Lat,data1.Long,data1.OFFENSE_CODE_GROUP):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(incidents2)

boston_map.add_child(incidents2)

boston_map
data1=data[data['YEAR']==2017][0:2000]

filename="Crime2017"

boston_map=folium.Map([Lat,Lon],zoom_start=12)

incidents2=plugins.MarkerCluster().add_to(boston_map)

for lat,lon,label in zip(data1.Lat,data1.Long,data1.OFFENSE_CODE_GROUP):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(incidents2)

boston_map.add_child(incidents2)

boston_map
data1=data[data['YEAR']==2018][0:2000]

filename="Crime2018"

boston_map=folium.Map([Lat,Lon],zoom_start=12)

incidents2=plugins.MarkerCluster().add_to(boston_map)

for lat,lon,label in zip(data1.Lat,data1.Long,data1.OFFENSE_CODE_GROUP):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(incidents2)

boston_map.add_child(incidents2)

boston_map