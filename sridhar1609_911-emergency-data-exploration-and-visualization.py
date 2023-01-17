# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import calendar
import folium
from mpl_toolkits.basemap import Basemap
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df911 = pd.read_csv("../input/911.csv")
df911 = df911.drop(columns = "e")
print(df911.head())
print(type(df911))
print(df911.info())
print(df911.dtypes)
lat = df911['lat'].values
lon = df911['lng'].values

# 1. Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='l', 
            lat_0=39, lon_0=-79,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter lat and long values

m.scatter(lon, lat, latlon=True,
          cmap='Reds', alpha=0.5)
plt.show()
print("Number of unique ZIP codes:",df911['zip'].nunique())
print("Number of unique township:",df911['twp'].nunique())
print("Number of unique title:",df911['title'].nunique())
df911["title"].value_counts()
#df = df911[df911["title"].str.match("EMS")]
print(df911["title"][df911["title"].str.match("EMS")].value_counts())
print(df911["title"][df911["title"].str.match("Traffic")].value_counts())
print(df911["title"][df911["title"].str.match("Fire")].value_counts())
df911["type"] = df911["title"].apply(lambda x: x.split(':')[0])
print(df911["type"].value_counts())
#df911["type1"] = df911["title"].apply(lambda x: x.split(':')[1])
df911["type"].value_counts().plot(fontsize = 18,
                                       kind = 'pie',
                                      autopct = "%1.0f%%",
                                     colors = ['#ff9999','#66b3ff','#99ff99'],
                                   )
plt.title("Distribution of Emergency category", fontsize=30)
plt.show()
df911['zip'].value_counts().head(10).plot.bar(color = 'chocolate')
plt.xlabel('Zip Codes',labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Zip Codes with Most Calls')
plt.show()
df911['twp'].value_counts().head(10).plot.bar(color = 'gold')
plt.xlabel('Townships', labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Townships with Most Calls')
plt.show()
plt.figure(figsize=(18,6))
sns.countplot( x='twp',data=df911,order=df911['twp'].value_counts().index[:10], hue='type', palette='rocket')
plt.title('Township wise type of calls')
plt.show()
plt.figure(figsize=(18,6))
sns.countplot( x='zip',data=df911,order=df911['zip'].value_counts().index[:10], hue='type', palette='rocket')
plt.title('ZIP wise type of calls')
plt.show()
print("EMS categories count",pd.unique(df911["title"][df911["title"].str.match("EMS")]).size)
print("Traffic categories count",pd.unique(df911["title"][df911["title"].str.match("Traffic")]).size)
print("Fire categories count",pd.unique(df911["title"][df911["title"].str.match("Fire")]).size)
df911["timeStamp"] = pd.to_datetime(df911["timeStamp"])
df911['Hour'] = df911['timeStamp'].apply(lambda x: x.hour)
df911['Month'] = df911['timeStamp'].apply(lambda x: x.month)
df911['Day of Week'] = df911['timeStamp'].apply(lambda x: x.dayofweek)
dmap= {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
dmonth = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
df911['Month']= df911['Month'].map(dmonth)
df911['Day of Week']= df911['Day of Week'].map(dmap)
df911.head()
worder = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
plt.figure(figsize=(12,8))
sns.countplot(x='Day of Week',data=df911,hue="type",order = worder,palette = 'viridis')
plt.legend(loc=[0,1])
plt.title('Day wise count plot for different types')
plt.show()
morder = ['January','February','March','April','May','June','July','August','September','October','November','December']
plt.figure(figsize=(12,8))
sns.countplot(x='Month',data=df911,hue="type",order = morder,palette = 'deep') 
plt.legend(loc=[0,1])
plt.title('Month wise count plot for different types')
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(x='Hour',data=df911,hue="type",palette = 'bright' )
plt.legend(loc=[0,1])
plt.title('Hour wise count plot for different types')
plt.show()
plt.xlabel('EMS Category')
plt.ylabel('Count')
plt.title('Count of Top 10 Emergencies under EMS Cataegory')
df911["title"][df911["title"].str.match("EMS")].value_counts().sort_values(ascending=False).head(10).plot.bar(color = 'salmon')
plt.show()
plt.xlabel('Fire Category')
plt.ylabel('Count')
plt.title('Count of Top 10 Emergencies under Fire Cataegory')
df911["title"][df911["title"].str.match("Fire")].value_counts().sort_values(ascending=False).head(10).plot.bar(color = 'sienna')
plt.show()
plt.xlabel('Traffic Category')
plt.ylabel('Count')
plt.title('Count of Top 5 Emergencies under Traffic Cataegory')
df911["title"][df911["title"].str.match("Traffic")].value_counts().sort_values(ascending=False).head().plot.bar(color = 'turquoise')
plt.show()
df911['Date'] = df911['timeStamp'].apply(lambda time:time.date())
plt.figure(figsize=(15,6))
plt.title('Traffic')
plt.ylabel('Number of Calls')
df911[df911['type'] == 'Traffic'].groupby('Date').count()['twp'].plot()
plt.tight_layout
plt.show()
plt.figure(figsize=(15,6))
plt.title('Fire')
plt.ylabel('Number of Calls')
df911[df911['type'] == 'Fire'].groupby('Date').count()['lat'].plot(color='green')
plt.tight_layout
plt.show()
plt.figure(figsize=(15,6))
plt.title('EMS')
df911[df911['type'] == 'EMS'].groupby('Date').count()['lat'].plot(color='teal')
plt.tight_layout
plt.show()
df_heatHour = df911.groupby(by = ['Day of Week', 'Hour']).count()['type'].unstack()
df_heatHour.index = pd.CategoricalIndex(df_heatHour.index, categories=worder)
df_heatHour.sort_index(level=0, inplace=True)
df_heatHour.head()
plt.figure(figsize=(10,7))
sns.heatmap(df_heatHour, cmap='viridis')
plt.title('Relationship of calls between Hour and Days of the week')
plt.show()
m_p = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',
       8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
df_heat_Month = df911.groupby(by = ['Day of Week', 'Month']).count()['type'].unstack()
df_heat_Month.index = pd.CategoricalIndex(df_heat_Month.index,categories = worder)
df_heat_Month.sort_index(level=0, inplace=True)
df_heat_Month.rename(columns = m_p,inplace=True)
df_heat_Month.head()
plt.figure(figsize=(10,5))
sns.heatmap(df_heat_Month, cmap='viridis')
plt.xlabel('Month')
plt.title('Relationship of calls between Month and Day of the week')
plt.show()
sns.clustermap(df_heat_Month)
plt.show()
locations = df911[['lat', 'lng']]
locationlist = locations.values.tolist()
print(len(locationlist))
print(locationlist[1])
map = folium.Map(location=[40.2172859, -75.405182], zoom_start=12)
folium.Marker(locationlist[1], popup=df911['title'][1]).add_to(map)
map
#map = folium.Map(location=[40.2172859, -75.405182], zoom_start=12)
#for point in range(0, len(locationlist)):
#    folium.Marker(locationlist[point], popup=df911['title'][point]).add_to(map)
#map    