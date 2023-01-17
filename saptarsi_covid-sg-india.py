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
import plotly.express as px
# Any results you write to the current directory are saved as output.
import plotly.graph_objs as go
import pandas as pd
df=pd.read_csv('/kaggle/input/covidcount/Daily.csv')
df.head(5)
Country=['India','SouthKorea']
lines = df.plot.line(x='Day', y=Country,title="First 32 days")
Country=['India','UK','Italy','Ireland','Germany','Spain','Austria']
lines = df.plot.line(x='Day', y=Country,title="First 32 days")
Country=['Iran','Iraq','India']
lines = df.plot.line(x='Day', y=Country,title="First 32 days")
import folium
from folium.plugins import HeatMap, HeatMapWithTime
%matplotlib inline
citylat=pd.read_csv('/kaggle/input/citylong/City Long Lat4th.csv')
citylat.head(5)
ind_geo_data='/kaggle/input/indiastates/states2.json'
m = folium.Map(location=[20, 78], zoom_start=4)

folium.Choropleth(
    geo_data=ind_geo_data,
    name='Confirmed cases - regions',
    key_on='feature.properties.name',
    fill_color='YlGn',
    fill_opacity=0.05,
    line_opacity=0.3,
).add_to(m)

radius_min = 2
radius_max = 40
weight = 1
fill_opacity = 0.2

_color_conf = 'red'
group0 = folium.FeatureGroup(name='<span style=\\"color: #EFEFE8FF;\\">Confirmed cases</span>')
for i in range(len(citylat)):
    lat = citylat.loc[i, 'Lat']
    lon = citylat.loc[i, 'Long']
    Active = citylat.loc[i, 'Count2']

    _radius_conf = np.sqrt(citylat.loc[i, 'Count2'])
    if _radius_conf < radius_min:
        _radius_conf = radius_min

    if _radius_conf > radius_max:
        _radius_conf = radius_max

    #_popup_conf = str(province) + '\n(Confirmed='+str(filtered_data_last.loc[i, 'Confirmed']) + '\nDeaths=' + str(death) + '\nRecovered=' + str(recovered) + ')'
    folium.CircleMarker(location = [lat,lon], 
                        radius = _radius_conf, 
                        color = _color_conf, 
                        fill_opacity = fill_opacity,
                        weight = weight, 
                        fill = True, 
                        fillColor = _color_conf).add_to(group0)

group0.add_to(m)
folium.LayerControl().add_to(m)
m
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary ()
map.drawcountries ()
map.drawcoastlines ()

lg=array(citylat['Long'])
lt=array(citylat['Lat'])
pt=array(citylat['Count2'])
nc=array(citylat['City'])

x, y = map(lg, lt)
city_sizes = citylat['Count2']
plt.scatter(x, y, s=city_sizes, marker="o", c=city_sizes, cmap=cm.Dark2, alpha=0.7)


plt.scatter(x, y, s=city_sizes, marker="o", c=city_sizes, cmap=cm.Dark2, alpha=0.7)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=8, fontweight='bold')

plt.title('Top Cities by Corona',fontsize=20)
df=pd.read_csv('/kaggle/input/dailystate/CoronaState.csv')
State=['Telengana','Maharastra','West Bengal','Kerala','Rajasthan','UP','Karnaraka','Gujrat']
lines = df.plot.line(x='Day', y=State,title="Last 20 days cumilative taking starting point as 0")
Cases=pd.read_csv('/kaggle/input/indiacorona/StateUT.csv')
Cases.rename(columns={"Confiremd": "Confirmed"},inplace=True)
Cases.head(5)
Cases = Cases.sort_values(['Confirmed/mn'], ascending = False).reset_index(drop=True)
Cases.drop(columns = ['Active', 'Recovered', 'Deceased','Population','Active/mn','Recovered/mn','Deceased/mn','ISO']).head(10).style.background_gradient(cmap='Reds')
Cases = Cases.sort_values(['Deceased/mn'], ascending = False).reset_index(drop=True)
Cases.drop(columns = ['Active', 'Recovered', 'Confirmed','Population','Active/mn','Recovered/mn','Confirmed/mn','ISO']).head(10).style.background_gradient(cmap='Reds')
Cases = Cases.sort_values(['Recovered/mn'], ascending = False).reset_index(drop=True)
Cases.drop(columns = ['Active', 'Deceased', 'Confirmed','Population','Active/mn','Deceased/mn','Confirmed/mn','ISO']).head(10).style.background_gradient(cmap='Greens')