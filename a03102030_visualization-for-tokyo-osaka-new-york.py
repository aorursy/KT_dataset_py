import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os 

import seaborn as sns

import pylab as pl

import geopandas as gpd

import folium

from folium import plugins



print(os.listdir("../input/city-lines/"))
cities=pd.read_csv("../input/city-lines/cities.csv")

stations=pd.read_csv("../input/city-lines/stations.csv")

tracks=pd.read_csv("../input/city-lines/tracks.csv")

lines=pd.read_csv("../input/city-lines/lines.csv")

track_lines=pd.read_csv("../input/city-lines/track_lines.csv")

print("cities size:",len(cities))

print("stations size:",len(stations))

print("tracks size:",len(tracks))

print("lines size:",len(lines))

print("track_lines size:",len(track_lines))
stations=stations.dropna(subset=['closure','name','opening'])

stations=stations[stations.closure>=9999]

stations=stations[stations.opening>0]

stations=stations[stations.opening<=2030]

stations.columns=['id','stations_name','geometry','buildstart','opening','closure','city_id']

stations['Long']=stations['geometry'].apply(lambda x: x.split('POINT(')[1].split(' ')[0])

stations['Lat']=stations['geometry'].apply(lambda x: x.split('POINT(')[1].split(' ')[1].split(')')[0])

id_country=pd.DataFrame({'city_id':cities.id,'country':cities.country,'name':cities.name})



stations=pd.merge(stations,id_country)

stations.head()
plt.figure(figsize=(18, 6))

plt.subplot(1, 2, 1) 

stations.name.value_counts()[:10].sort_values().plot.barh()

plt.title("Top 10 city by stations",size=18)

plt.xlabel("stations")

plt.subplot(1, 2, 2) 

stations.country.value_counts()[:10].sort_values().plot.barh()

plt.title("Top 10 country by stations",size=18)

plt.xlabel("stations")
fig,ax=plt.subplots(figsize=(10,6))

stations.groupby(['opening','country'])['stations_name'].agg('count').unstack().plot(ax=ax)

plt.legend(loc=0, bbox_to_anchor=(1.05,1.1))

plt.xlabel('')

plt.ylabel('stations')

plt.title("The number of opening stations by year (all country)",size=18)
plt.figure(figsize=(35,55))

for i in range(len(stations.country.unique())):

    plt.subplot(9,3,i+1)

    stations[stations.country==stations.country.unique()[i]].name.value_counts().sort_values().plot.barh()

    plt.xlabel('stations')

    plt.title("The number of stations in "+stations.country.unique()[i],size=20)

map_all=folium.Map()

stations_new=pd.DataFrame({"Lat":stations['Lat'],"Long":stations['Long']})

map_all.add_child(plugins.HeatMap(data=stations_new))



map_all
tokyo_lines=lines[lines.city_id==114]

tokyo_track_lines=track_lines[track_lines.city_id==114]

tokyo_tracks=tracks[tracks.city_id==114].drop(columns=['buildstart','opening','closure','city_id'])

tokyo_tracks.columns=['section_id','geometry','length']

tokyo_track_lines=pd.merge(tokyo_track_lines,tokyo_tracks)

tokyo_track_lines=tokyo_track_lines.drop(columns=['id','created_at','updated_at','city_id'])

tokyo_track_lines.columns=['section_id','id','geometry','length']

tokyo_lines=pd.merge(tokyo_track_lines,tokyo_lines)

tokyo_stations=stations[stations['city_id']==114]

tokyo_stations.head()
x=[]

y=[]

z=[]

for i in range(len(tokyo_lines)):

    sp=tokyo_lines.iloc[i].geometry.split('(')[1].split(')')[0].split(',')

    for j in range(len(sp)):

        x.append(sp[j].split(' ')[0])

        y.append(sp[j].split(' ')[1])

        z.append(tokyo_lines.url_name[i])

fix=pd.DataFrame({'x':x,'y':y,'z':z})

fix['x']=fix['x'].astype(float)

fix['y']=fix['y'].astype(float)

plt.figure(figsize=(25, 25))

plt.subplot(2, 2, 1) 

ax=sns.scatterplot(x="x", y="y", hue="z",data=fix)

plt.legend(loc=0, bbox_to_anchor=(1.05,0.6))

plt.title("lines for Tokyo",size=20)

ax.get_legend().remove()

ax.set(xlabel='Longitude', ylabel='LATITUDE')

plt.subplot(2,2,2)

(tokyo_lines.groupby(['url_name'])['length'].sum()/1000).sort_values(ascending= False)[:10].sort_values().plot.barh()

plt.ylabel(' ')

plt.xlabel('length(km)')

plt.title("Top 10 track by length",size=20)

plt.subplot(2,2,3)

tokyo_stations.groupby(['opening'])['id'].agg('count').plot()

plt.xlabel(' ')

plt.ylabel('stations')

plt.title("Number of opening stations by year",size=20)

plt.subplot(2,2,4)

tokyo_lines.name.value_counts()[:10].sort_values().plot.barh()

plt.xlabel('counts')

plt.title("Top 10 line by number",size=20)


stations_tokyo_2000=stations[stations['city_id']==114][0:2000]

Long=139.75

Lat=35.67

tokyo_map=folium.Map([Lat,Long],zoom_start=12)



tokyo_stations_map=plugins.MarkerCluster().add_to(tokyo_map)

for lat,lon,label in zip(stations_tokyo_2000.Lat,stations_tokyo_2000.Long,stations_tokyo_2000.stations_name):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(tokyo_stations_map)

tokyo_map.add_child(tokyo_stations_map)



tokyo_map

osaka_lines=lines[lines.city_id==91]

osaka_track_lines=track_lines[track_lines.city_id==91]

osaka_tracks=tracks[tracks.city_id==91].drop(columns=['buildstart','opening','closure','city_id'])

osaka_tracks.columns=['section_id','geometry','length']

osaka_track_lines=pd.merge(osaka_track_lines,osaka_tracks)

osaka_track_lines=osaka_track_lines.drop(columns=['id','created_at','updated_at','city_id'])

osaka_track_lines.columns=['section_id','id','geometry','length']

osaka_lines=pd.merge(osaka_track_lines,osaka_lines)

osaka_stations=stations[stations['city_id']==91]

osaka_stations.head()
x=[]

y=[]

z=[]

for i in range(len(osaka_lines)):

    sp=osaka_lines.iloc[i].geometry.split('(')[1].split(')')[0].split(',')

    for j in range(len(sp)):

        x.append(sp[j].split(' ')[0])

        y.append(sp[j].split(' ')[1])

        z.append(osaka_lines.url_name[i])

fix=pd.DataFrame({'x':x,'y':y,'z':z})

fix['x']=fix['x'].astype(float)

fix['y']=fix['y'].astype(float)

plt.figure(figsize=(27, 27))

plt.subplot(2, 2, 1) 

ax=sns.scatterplot(x="x", y="y", hue="z",data=fix)

plt.legend(loc=0, bbox_to_anchor=(1.05,0.6))

plt.title("lines for osaka",size=20)

ax.get_legend().remove()

ax.set(xlabel='Longitude', ylabel='LATITUDE')

plt.subplot(2,2,2)

(osaka_lines.groupby(['url_name'])['length'].sum()/1000).sort_values(ascending= False)[:10].sort_values().plot.barh()

plt.ylabel(' ')

plt.xlabel('length(km)')

plt.title("Top 10 track by length",size=20)

plt.subplot(2,2,3)

osaka_stations.groupby(['opening'])['id'].agg('count').plot()

plt.xlabel(' ')

plt.ylabel('stations')

plt.title("Number of opening stations by year",size=20)

plt.subplot(2,2,4)

osaka_lines.name.value_counts()[:10].sort_values().plot.barh()

plt.xlabel('counts')

plt.title("Top 10 line by number",size=20)


stations_osaka=stations[stations['city_id']==91]

Long=135.5

Lat=34.53

osaka_map=folium.Map([Lat,Long],zoom_start=12)



osaka_stations_map=plugins.MarkerCluster().add_to(osaka_map)

for lat,lon,label in zip(stations_osaka.Lat,stations_osaka.Long,stations_osaka.stations_name):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(osaka_stations_map)

osaka_map.add_child(osaka_stations_map)



osaka_map
New_York_lines=lines[lines.city_id==206]

New_York_track_lines=track_lines[track_lines.city_id==206]

New_York_tracks=tracks[tracks.city_id==206].drop(columns=['buildstart','opening','closure','city_id'])

New_York_tracks.columns=['section_id','geometry','length']

New_York_track_lines=pd.merge(New_York_track_lines,New_York_tracks)

New_York_track_lines=New_York_track_lines.drop(columns=['id','created_at','updated_at','city_id'])

New_York_track_lines.columns=['section_id','id','geometry','length']

New_York_lines=pd.merge(New_York_track_lines,New_York_lines)

New_York_stations=stations[stations['city_id']==206]

New_York_stations.head()
x=[]

y=[]

z=[]

for i in range(len(New_York_lines)):

    sp=New_York_lines.iloc[i].geometry.split('(')[1].split(')')[0].split(',')

    for j in range(len(sp)):

        x.append(sp[j].split(' ')[0])

        y.append(sp[j].split(' ')[1])

        z.append(New_York_lines.url_name[i])

fix=pd.DataFrame({'x':x,'y':y,'z':z})

fix['x']=fix['x'].astype(float)

fix['y']=fix['y'].astype(float)

plt.figure(figsize=(27, 27))

plt.subplot(2, 2, 1) 

ax=sns.scatterplot(x="x", y="y", hue="z",data=fix)

plt.legend(loc=0, bbox_to_anchor=(1.05,0.6))

plt.title("lines for New_York",size=20)

ax.get_legend().remove()

ax.set(xlabel='Longitude', ylabel='LATITUDE')

plt.subplot(2,2,2)

(New_York_lines.groupby(['url_name'])['length'].sum()/1000).sort_values(ascending= False)[:10].sort_values().plot.barh()

plt.ylabel(' ')

plt.xlabel('length(km)')

plt.title("Top 10 track by length",size=20)

plt.subplot(2,2,3)

New_York_stations.groupby(['opening'])['id'].agg('count').plot()

plt.xlabel(' ')

plt.ylabel('stations')

plt.title("Number of opening stations by year",size=20)

plt.subplot(2,2,4)

New_York_lines.name.value_counts()[:10].sort_values().plot.barh()

plt.xlabel('counts')

plt.title("Top 10 line by number",size=20)
stations_New_York=stations[stations['city_id']==206]

Long=-73.97

Lat=40.78

New_York_map=folium.Map([Lat,Long],zoom_start=12)



New_York_stations_map=plugins.MarkerCluster().add_to(New_York_map)

for lat,lon,label in zip(stations_New_York.Lat,stations_New_York.Long,stations_New_York.stations_name):

    folium.Marker(location=[lat,lon],icon=None,popup=label).add_to(New_York_stations_map)

New_York_map.add_child(New_York_stations_map)



New_York_map