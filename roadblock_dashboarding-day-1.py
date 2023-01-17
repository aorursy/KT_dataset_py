import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import folium
import ast
from folium import plugins
#print(os.listdir("../input"))
data=pd.read_csv("../input/fire-department-calls-for-service.csv")
data.head()
s=data.groupby(['City'])['Supervisor District'].count()
plt.figure(figsize=(5,2.5))
s.plot.bar()
m = folium.Map(location=[37,-122], tiles="cartodbdark_matter", zoom_start=6)#cartodbdark_matter/Mapbox Bright
k=0
for i,city in zip(data['Location'],data['City']):
    #print(i+" "+city)
    my_dict=ast.literal_eval(i)
    #print(my_dict)
    folium.Marker(
      location=[float(my_dict['latitude']),float(my_dict['longitude'])],
      popup=city,
    ).add_to(m)
    if k==1000:
        break
    k+=1
m
s1=data.groupby(['Call Type'])['Number of Alarms'].count()
plt.figure(figsize=(5,2.5))
s1.plot.bar()
map_hooray = folium.Map(location=[37,-122],tiles="cartodbdark_matter",zoom_start = 6) 
k=0
place_list=[]
for i,city in zip(data['Location'],data['City']):
    my_dict=ast.literal_eval(i)
    place_list.append([float(my_dict['latitude']),float(my_dict['longitude'])])
    if k==1000:
        break
    k+=1
plugins.HeatMap(place_list).add_to(map_hooray)
map_hooray
priorities=data.groupby(['Priority'])['Number of Alarms'].count()
plt.figure(figsize=(5,5))
#plt.scatter(priorities)
priorities.plot.pie()
priorities['1']
priorities.plot.barh()
