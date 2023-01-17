import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import os

import geopandas as gpd

import folium

from folium import plugins

import datetime 

import re
os.listdir("../input/autotel-shared-car-locations")
data=pd.read_csv("../input/autotel-shared-car-locations/sample_table.csv")

data.head()

data=data[data['total_cars']>0]
data.isnull().sum()
data['date']=data.timestamp.apply(lambda x:x.split(' ')[0])

pd.to_datetime(data['date'])
plt.figure(figsize=(12,12))

y=data.groupby('date')['total_cars'].agg('sum').index

x=data.groupby('date')['total_cars'].agg('sum')

sns.barplot(x=x,y=y)

plt.title("total_car in the data by date",size=24)

plt.xlabel('cars')
index=data.groupby('date')['timestamp'].agg('count').index

plt.figure(figsize=(15,100))

for i in range(len(index)):

    plt.subplot(15,2,i+1)

    sns.scatterplot(x='latitude',y='longitude',alpha=0.01,data=data[data.date==index[i]])

    plt.title('total_car '+index[i],size=20)

    plt.xlim(32,32.17)

    plt.ylim(34.74,34.85)

    
data.carsList.apply(lambda x:x.strip('[]'))

data['carsList']=data.carsList.apply(lambda x:x.strip('[]'))

data.head()
data_37=data[data.carsList.apply(lambda x:re.match('37',x)!=None)]

data_37_new=data_37[data.date==data_37.date.unique()[0]].drop_duplicates(subset=['longitude'])

for i in range(1,len(data_37.date.unique())):

    data_37_concate=data_37[data.date==data_37.date.unique()[i]].drop_duplicates(subset=['longitude'])

    data_37_new=pd.concat([data_37_new,data_37_concate],axis=0)



Long=34.78

Lat=32.05

data_37_map=folium.Map([Lat,Long],zoom_start=12)



data_37_cars_map=plugins.MarkerCluster().add_to(data_37_map)

for lat,lon,label in zip(data_37_new.latitude,data_37_new.longitude,data_37_new.timestamp):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_37_cars_map)

data_37_map.add_child(data_37_cars_map)



data_37_map
data_37=data[data.carsList.apply(lambda x:re.match('37',x)!=None)]

data_37_new=data_37[data.date==data_37.date.unique()[0]].drop_duplicates(subset=['latitude'])

for i in range(1,len(data_37.date.unique())):

    data_37_concate=data_37[data.date==data_37.date.unique()[i]].drop_duplicates(subset=['latitude'])

    data_37_new=pd.concat([data_37_new,data_37_concate],axis=0)



Long=34.78

Lat=32.05

data_37_map=folium.Map([Lat,Long],zoom_start=12)



data_37_cars_map=plugins.MarkerCluster().add_to(data_37_map)

for lat,lon,label in zip(data_37_new.latitude,data_37_new.longitude,data_37_new.timestamp):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_37_cars_map)

data_37_map.add_child(data_37_cars_map)



data_37_map
data_total_cars_1=data[data.total_cars==1].drop_duplicates(subset=['latitude'])[:2000]

data_total_cars_1.head()
Long=34.78

Lat=32.05

data_total_cars_1_map=folium.Map([Lat,Long],zoom_start=12)

data_total_cars_1['label']=data_total_cars_1.apply(lambda x: (x['timestamp'],'car:'+str(x['carsList'])),axis=1)



data_total_cars_1_cars_map=plugins.MarkerCluster().add_to(data_total_cars_1_map)

for lat,lon,label in zip(data_total_cars_1.latitude,data_total_cars_1.longitude,data_total_cars_1.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_total_cars_1_cars_map)

data_total_cars_1_map.add_child(data_total_cars_1_cars_map)



data_total_cars_1_map
data_total_cars_2=data[data.total_cars==1].drop_duplicates(subset=['latitude'])[2000:4000]

Long=34.78

Lat=32.05

data_total_cars_2_map=folium.Map([Lat,Long],zoom_start=12)

data_total_cars_2['label']=data_total_cars_2.apply(lambda x: (x['timestamp'],'car:'+str(x['carsList'])),axis=1)



data_total_cars_2_cars_map=plugins.MarkerCluster().add_to(data_total_cars_2_map)

for lat,lon,label in zip(data_total_cars_2.latitude,data_total_cars_2.longitude,data_total_cars_2.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_total_cars_2_cars_map)

data_total_cars_2_map.add_child(data_total_cars_2_cars_map)



data_total_cars_2_map
data_total_cars_3=data[data.total_cars==1].drop_duplicates(subset=['latitude'])[4000:6000]

Long=34.78

Lat=32.05

data_total_cars_3_map=folium.Map([Lat,Long],zoom_start=12)

data_total_cars_3['label']=data_total_cars_3.apply(lambda x: (x['timestamp'],'car:'+str(x['carsList'])),axis=1)



data_total_cars_3_cars_map=plugins.MarkerCluster().add_to(data_total_cars_3_map)

for lat,lon,label in zip(data_total_cars_3.latitude,data_total_cars_3.longitude,data_total_cars_3.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_total_cars_3_cars_map)

data_total_cars_3_map.add_child(data_total_cars_3_cars_map)



data_total_cars_3_map
data_total_cars_4=data[data.total_cars==1].drop_duplicates(subset=['latitude'])[6000:8000]

Long=34.78

Lat=32.05

data_total_cars_4_map=folium.Map([Lat,Long],zoom_start=12)

data_total_cars_4['label']=data_total_cars_4.apply(lambda x: (x['timestamp'],'car:'+str(x['carsList'])),axis=1)



data_total_cars_4_cars_map=plugins.MarkerCluster().add_to(data_total_cars_4_map)

for lat,lon,label in zip(data_total_cars_4.latitude,data_total_cars_4.longitude,data_total_cars_4.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_total_cars_4_cars_map)

data_total_cars_4_map.add_child(data_total_cars_4_cars_map)



data_total_cars_4_map
data_total_cars_5=data[data.total_cars==1].drop_duplicates(subset=['latitude'])[8000:-1]

Long=34.78

Lat=32.05

data_total_cars_5_map=folium.Map([Lat,Long],zoom_start=12)

data_total_cars_5['label']=data_total_cars_5.apply(lambda x: (x['timestamp'],'car:'+str(x['carsList'])),axis=1)



data_total_cars_5_cars_map=plugins.MarkerCluster().add_to(data_total_cars_5_map)

for lat,lon,label in zip(data_total_cars_5.latitude,data_total_cars_5.longitude,data_total_cars_5.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_total_cars_5_cars_map)

data_total_cars_5_map.add_child(data_total_cars_5_cars_map)



data_total_cars_5_map
data_total_cars_6=data[data.total_cars==2].drop_duplicates(subset=['latitude'])

data_total_cars_6.head()

Long=34.78

Lat=32.05

data_total_cars_6_map=folium.Map([Lat,Long],zoom_start=12)

data_total_cars_6['label']=data_total_cars_6.apply(lambda x: (x['timestamp'],'car:'+str(x['carsList'])),axis=1)



data_total_cars_6_cars_map=plugins.MarkerCluster().add_to(data_total_cars_6_map)

for lat,lon,label in zip(data_total_cars_6.latitude,data_total_cars_6.longitude,data_total_cars_6.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_total_cars_6_cars_map)

data_total_cars_6_map.add_child(data_total_cars_6_cars_map)



data_total_cars_6_map
data_total_cars_7=data[data.total_cars>=3].drop_duplicates(subset=['latitude'])

data_total_cars_7.head()
Long=34.78

Lat=32.05

data_total_cars_7_map=folium.Map([Lat,Long],zoom_start=12)

data_total_cars_7['label']=data_total_cars_7.apply(lambda x: (x['timestamp'],'car:'+str(x['carsList'])),axis=1)



data_total_cars_7_cars_map=plugins.MarkerCluster().add_to(data_total_cars_7_map)

for lat,lon,label in zip(data_total_cars_7.latitude,data_total_cars_7.longitude,data_total_cars_7.label):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(data_total_cars_7_cars_map)

data_total_cars_7_map.add_child(data_total_cars_7_cars_map)



data_total_cars_7_map