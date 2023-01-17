# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.head()
!pip install gmplot
import gmplot as gp
google_api_key = "AIzaSyAPIfruYAS0bFf2WGXKBMgtCtWyd_AjaZ8"
gmap1 = gp.GoogleMapPlotter(40.7128, 
                                -74.0060, 13 )
gmap1.apikey = google_api_key
# gmap1.draw("map1.html")
import matplotlib.pyplot as plt
data['price'].plot(figsize=(20,16))
plt.legend()
plt.show()
data.columns
data.describe()
data.dtypes
data.corr().style.background_gradient(cmap='coolwarm')
########################
# 残缺数据调查
########################
data.isnull().any()
data.isnull().any(1)
null_columns = data.columns[data.isnull().any()]
print(data[null_columns].isnull().sum())
# 残缺率
total_ele = data.size
nonempty_ele = data.notnull().sum().sum()
print("total number: %d" % total_ele)
print("total non-empty element number: %d" % nonempty_ele)
print("Empty element ratio: %.2f%%" % ((1 - nonempty_ele / total_ele) * 100))
data['last_review'].notnull()
data[data['last_review'].isnull()]
len(data[data['last_review'].isnull() & data['reviews_per_month'].isnull()])
data[data['price'] == 0]
len(data[data['price'] == 0])
# 残缺数据补足
data_filled = data.fillna(value={'reviews_per_month': 0})
data_filled
data_filled[data_filled['name'].isnull()]
data_filled[data_filled['host_name'].isnull()]
data_clean = data_filled[(data_filled['host_name'].notnull()) & (data_filled['price'] > 0)]
data_clean
data_clean.isnull().any()
print(data_clean.isnull().sum())
########################
# 房源分布
########################
data = data_clean
# 房源随地区的分布
data.groupby('neighbourhood_group')['id'].count().sort_values(ascending=False)
data.groupby('neighbourhood_group')['id'].count().sort_values(ascending=False).plot.bar()
# 转换成dict
data.groupby('neighbourhood_group')['id'].count().sort_values(ascending=False).to_dict()
# 房源随地区的分布
data.groupby('room_type')['id'].count().sort_values(ascending=False)
data.groupby('room_type')['id'].count().sort_values(ascending=False) / data['id'].count()
data.groupby('room_type')['id'].count().sort_values(ascending=False).plot.bar()
(data.groupby('room_type')['id'].count().sort_values(ascending=False) / data['id'].count()).plot.bar(figsize=(10,8))
# 房源随租金的分布
data['price'].describe()
data.groupby('price')['id'].count().head(100)
price_count = np.array(list(data.groupby('price')['id'].count().to_dict().items()))
pc_data = pd.DataFrame(price_count, columns=['price', 'count'])
pc_data
pc_data[pc_data['price'] <= 500].plot.bar(x='price', y='count', figsize=(20,16))
data.groupby('price')['id'].count().plot.bar(figsize=(20,16))
data[data['price'] <= 500].groupby('price')['id'].count().plot.bar(figsize=(20,16))
data['price'].hist()
data['price'].hist(bins = 100, range=(0, 500), figsize=(20,16))
len(data[data['minimum_nights'] <= 200])
data['minimum_nights'].hist(bins = 100, range=(0, 200), figsize=(20,16))
########################
# 热门度分析
########################
# 1. 热门房东排名
data.groupby('host_name')['id'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(20,16))
data.groupby('host_id')['id'].count().sort_values(ascending=False).head(10).plot.bar(figsize=(20,16))
data.groupby('host_id')['id'].count().sort_values(ascending=False)
id_name = dict(data.groupby('host_id')[['host_id', 'host_name']].head(1).values)
id_name[219517861]
host_ids, count = zip(*(data.groupby('host_id')['id'].count().sort_values(ascending=False).head(10).to_dict().items()))
host_ids
host_names = [id_name[id] for id in host_ids]
plt.figure(figsize=(20,16))
plt.bar(host_names, count)
plt.show()
########################
# 租金与租房特性的相关性分析
########################
data.groupby('room_type')['price'].describe()
data.groupby('neighbourhood_group')['price'].describe()
########################
# 地图可视化
########################
plt.figure(figsize=(20,16))
nyc_img=plt.imread("../input/new-york-city-airbnb-open-data/New_York_City_.png", 0)
plt.imshow(nyc_img,zorder=-10,extent=[-74.25, -73.690, 40.487, 40.925])
data[(data['price'] <= 500) & (data['room_type'] == 'Private room')].plot(kind='scatter', x='longitude', y='latitude', label='price', c='price',
            cmap=plt.get_cmap('coolwarm'), colorbar=True, alpha=0.4, figsize=(20,16))
plt.legend()
#initializing the figure size
plt.figure(figsize=(20,16))
nyc_img=plt.imread("../input/new-york-city-airbnb-open-data/New_York_City_.png", 0)
#scaling the image based on the latitude and longitude max and mins for proper output
plt.imshow(nyc_img,zorder=-10,extent=[-74.25, -73.690, 40.487, 40.925])
ax=plt.gca()
#using scatterplot again
data[(data['price'] <= 500) & (data['neighbourhood_group'] == 'Brooklyn')].plot(kind='scatter', x='longitude', y='latitude', label='price', c='price', ax=ax, 
           cmap=plt.get_cmap('coolwarm'), colorbar=True, alpha=0.4, zorder=5)

plt.legend()
plt.show()
import folium
from folium.plugins import HeatMap
m=folium.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(data[['latitude','longitude']],radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
display(m)
from folium.plugins import MarkerCluster
from folium import plugins
print('Top 1000 Expensive Prive Rooms')
Long=-73.80
Lat=40.80
private_room_data = data[data['room_type'] == 'Private room'].sort_values(by=['price'], ascending=True).head(1000)
mapdf1=folium.Map([Lat,Long],zoom_start=10,)

mapdf1_rooms_map=MarkerCluster().add_to(mapdf1)

for lat,lon,label in zip(private_room_data.latitude, private_room_data.longitude, private_room_data.name):
    folium.Marker(location=[lat,lon],icon=folium.Icon(icon='home'),popup=label).add_to(mapdf1_rooms_map)
mapdf1.add_child(mapdf1_rooms_map)

mapdf1
import branca
data['neighbourhood_group'].unique()
import branca.colormap as cmp
Long=-73.80
Lat=40.80
private_room_data = data[data['room_type'] == 'Private room'].sort_values(by=['price'], ascending=False).head(300)
prd = private_room_data
linear = cmp.LinearColormap(
    ['white', 'black'],
    vmin=np.log(prd['price'].min()), vmax=np.log(prd['price'].max())
)

mapdf1=folium.Map([Lat,Long],zoom_start=10)

for lat, lon, price, neighbourhood_group, name in zip(prd.latitude, prd.longitude, prd.price, prd.neighbourhood_group, prd.name):
    folium.Marker(location=[lat,lon], icon=folium.Icon(color='blue', icon_color=linear(np.log(price)), icon='home'), popup="%s\t$%d"%(name, price)).add_to(mapdf1)
mapdf1
import branca.colormap as cmp
Long=-73.80
Lat=40.80
private_room_data = data[data['room_type'] == 'Private room'].sort_values(by=['price'], ascending=False).head(300)
prd = private_room_data
linear = cmp.LinearColormap(
    ['white', 'black'],
    vmin=np.log(prd['price'].min()), vmax=np.log(prd['price'].max())
)

color_dict = {
    'Brooklyn': 'blue', 'Manhattan': 'red', 'Queens': 'green', 'Staten Island': 'yellow', 'Bronx': 'purple'
}

mapdf1=folium.Map([Lat,Long],zoom_start=10)

for lat, lon, price, neighbourhood_group, name in zip(prd.latitude, prd.longitude, prd.price, prd.neighbourhood_group, prd.name):
    folium.Marker(location=[lat,lon], icon=folium.Icon(color=color_dict[neighbourhood_group], icon_color=linear(np.log(price)), icon='home'), popup="%s\t$%d"%(name, price)).add_to(mapdf1)
mapdf1
data['room_type'].unique()
import branca.colormap as cmp
Long=-73.80
Lat=40.80
private_room_data = data.sort_values(by=['price'], ascending=False).head(300)
prd = private_room_data
linear = cmp.LinearColormap(
    ['white', 'black'],
    vmin=np.log(prd['price'].min()), vmax=np.log(prd['price'].max())
)

color_dict = {
    'Private room': 'blue', 'Entire home/apt': 'red', 'Shared room': 'green'
}

mapdf1=folium.Map([Lat,Long],zoom_start=10)

for lat, lon, price, room_type, name in zip(prd.latitude, prd.longitude, prd.price, prd.room_type, prd.name):
    folium.Marker(location=[lat,lon], icon=folium.Icon(color=color_dict[room_type], icon_color=linear(np.log(price)), icon='home'), popup="%s\t$%d"%(name, price)).add_to(mapdf1)
mapdf1
