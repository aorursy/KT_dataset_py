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



#install updates for python visualisation and folium graph object



#!pip install git+https://github.com/python-visualization/branca

#!pip install git+https://github.com/sknzl/folium@update-css-url-to-https





#import python graph objects from Kaggle library



import matplotlib.pyplot as plt #plotting

import seaborn as sns #for beatiful visualization

import folium

from folium import plugins





#import Australia Bush Fire data file; NASA Modis file



aus_fire_arch_m6 = pd.read_csv("../input/ausbushfirearchivem6modisnasa/fire_archive_M6_146074_WIP.csv")

type(aus_fire_arch_m6)





dataf = aus_fire_arch_m6

dataf.head()



dataf.info()





#filter the data



dt_filter = dataf.filter(["latitude","longitude","acq_date","frp"])

dt_filter.head()





#areas with high fire radiation



dataf_highimpact = dt_filter.sort_values(by='frp',ascending=False).head(10)

dataf_highimpact



#Create a map

mapF = folium.Map(location=[-35.0,144], control_scale=True, zoom_start=3,attr = "text some")

dataf2 = dataf_highimpact.copy()



# loop through data to create Marker for each fire spots

for i in range(0,len(dataf2)):

    

    folium.Marker(

    location=[dataf2.iloc[i]['latitude'], dataf2.iloc[i]['longitude']],

    

    #popup=popup,

    tooltip="frp: " + str(dataf2.iloc[i]['frp']) + "<br/> date: "+ str(dataf2.iloc[i]['acq_date']),

    icon=folium.Icon(color='red',icon='fire',prefix="fa"),

    ).add_to(mapF)

        

mapF

#mapF.to_csv('BushFireMap.csv', index = False)

#incorporating the date attribute into the frp map



dataf2_date = dt_filter[['acq_date','frp']].set_index('acq_date')

dataf2_date_top = dataf2_date.groupby('acq_date').sum().sort_values(by='frp',ascending=False)

#dataf2_date_top.head(10)







plt.figure(figsize=(15,5))

sns.set_palette("pastel")

ax = sns.barplot(x='acq_date',y='frp',data=dt_filter)

for ind, label in enumerate(ax.get_xticklabels()):

    if ind % 3 == 0:  # every 2nd label is kept

        label.set_visible(True)

    else:

        label.set_visible(False)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.xlabel("Date")

plt.ylabel('FRP (fire radiation power)')

plt.title("time line of bushfire in Australia")

plt.tight_layout()
from folium.plugins import HeatMapWithTime

# A function to get heat map with time given in the dataset



def getmap(input_data,location,zoom,radius):

    

    #get day list

    dt_filter_map = input_data[['acq_date','latitude','longitude','frp']]

    dt_filter_day_list = []

    for day in dt_filter_map.acq_date.sort_values().unique():

        dt_filter_day_list.append(dt_filter_map.loc[dt_filter_map.acq_date == day, ['acq_date','latitude', 'longitude', 'frp']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist())

    

    # Create a map using folium

    mapF2 = folium.Map(location, zoom_start = zoom,tiles = 'Stamen Terrain')

    #creating heatmap with time

    HeatMapWithTime(dt_filter_day_list,index = list(dt_filter_map.acq_date.sort_values().unique()), auto_play=False,radius=radius, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(mapF2)



    return mapF2

getmap(dt_filter,[-27,132],3.5,3)
dataf3 = dataf

dataf3.head()
#filter by the 'confidence' column where value is >= 50



df3_mod1 = dataf3[dataf3['confidence'] >= 50]

df3_mod1.head()

#df3_mod1.shape

#dataf3.shape
#filter by the 'frp' column where value is >= 40 :"40 = level of fire intensity (fire radiative power, frp)"



df3_mod2 = df3_mod1[df3_mod1['frp'] >= 40]

df3_mod2.head()

#df3_mod2.shape

#dataf3.shape
#filter by the 'type' column where value is == 0 :"0 = vegetation fire"



df3_mod3 = df3_mod2[df3_mod2['type'] == 0]

df3_mod3.head()

#df3_mod3.shape

#dataf3.shape
#filter by the 'daynight' column where value is == 'D' :"D = fire detected during the day"



df3_mod4 = df3_mod3[df3_mod3['daynight'] == 'D']

df3_mod4.head()

#df3_mod4.shape

#dataf3.shape
#filter by the 'satellite' column where value is == 'Aqua' :"Aqua = fire detected by MODIS Aqua satellite. 

#The Aqua satellite travels from south to north"



df3_mod5 = df3_mod4[df3_mod4['satellite'] == 'Aqua']

df3_mod5.head()

#df3_mod5.shape

#dataf3.shape
#filter the data to only four attributes



df3_mod5_filter = df3_mod5.filter(["latitude","longitude","acq_date","frp"])

df3_mod5_filter.head()

#df3_mod5_filter.shape
#sort the filtered data by 'acq_date' in descending order



#df3_mod5_sort = df3_mod5_filter.sort_values(by='acq_date',ascending=False).head(30)

df3_mod5_filter2 = df3_mod5_filter[df3_mod5_filter['acq_date'] >= '2019-10-01']

df3_mod5_filter2.head(10)

#df3_mod5_filter2.shape
#Create a map

mapF3 = folium.Map(location = [-35.0,144], control_scale = True, zoom_start = 3,attr = "text some")

df3_mapdt = df3_mod5_filter2.copy()

df3_mapdt.info()



# loop through data to create Marker for each fire spots

for i in range(0,len(df3_mapdt)):

    

    folium.Marker(

    location=[df3_mapdt.iloc[i]['latitude'], df3_mapdt.iloc[i]['longitude']],

    

    #popup=popup,

    tooltip = "frp: " + str(df3_mapdt.iloc[i]['frp']) + "<br/> date: "+ str(df3_mapdt.iloc[i]['acq_date']),

    icon = folium.Icon(color = 'red',icon = 'fire',prefix = "fa"),

    ).add_to(mapF3)

        

mapF3
df3_mod5_filter3 = df3_mod5_filter2.filter(['latitude','longitude','frp'])

df3_mod5_filter3.head()

#df3_mod5_filter3.shape
from sklearn.cluster import KMeans



df3_mapdt2 = df3_mod5_filter3.copy()

  

kmeans = KMeans(n_clusters = 50).fit(df3_mapdt2)

centroids = kmeans.cluster_centers_

print(centroids)



plt.scatter(df3_mapdt2['latitude'], df3_mapdt2['longitude'], c = kmeans.labels_.astype(float), s = 50, alpha = 0.5)

plt.scatter(centroids[:, 0], centroids[:, 1], c = 'red', s = 50)

plt.show()
#df_dum = pd.DataFrame(centroids, index = ['pt1','pt2','pt3','pt4','pt5','pt6','pt7','pt8','pt9','pt10'], columns = ['lat','long','frp2'])

df_dum = pd.DataFrame(centroids, columns = ['latitude2','longitude2','frp2'])

df_dum
#Create a map for the predicted fire locations



mapF4 = folium.Map(location = [-35.0,144], control_scale = True, zoom_start = 3,attr = "text some")

df4_mapdt = df_dum.copy()

df4_mapdt.info()



# loop through data to create Marker for each fire spots

for i in range(0,len(df4_mapdt)):

    

    folium.Marker(

    location=[df4_mapdt.iloc[i]['latitude2'], df4_mapdt.iloc[i]['longitude2']],

    

    #popup=popup,

    tooltip = "frp: " + str(df4_mapdt.iloc[i]['frp2']) + "<br/> date: "+ str("no date"),

    icon = folium.Icon(color = 'red',icon = 'fire',prefix = "fa"),

    ).add_to(mapF4)

        

mapF4