import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import geopandas as gpd

from geopandas import GeoDataFrame

from shapely.geometry import Point,Polygon



import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster



import requests 
URL = "https://coronavirus-tracker-api.herokuapp.com/v2/locations"
# sending get request and saving the response as response object 

r = requests.get(url = URL) 



# extracting data in json format 

data = r.json() 
coordintes_confirmed = []

for values in data['locations']:

    country = values['country']

    latitude = float((values['coordinates'])['latitude'])

    longitude = float((values['coordinates'])['longitude'])

    confirmed = int((values['latest'])['confirmed'])

    deaths = int((values['latest'])['deaths'])

    if(values['province'] != ""):

        province = values['province']

    else:

        province = 'Nan'

    coordintes_confirmed.append((country,latitude,longitude,province,confirmed,deaths))
world_frame = pd.DataFrame(coordintes_confirmed, columns=['Country', 'Latitude', 'Longitude', 'Province','Confirmed_Cases','Deaths'])
world_frame.head()
world_frame_map = world_frame.copy()
world_frame_map.head()
world_frame_map_copy = world_frame_map.copy()

geometry_copy = [Point(xy) for xy in zip(world_frame_map.Latitude, world_frame_map.Longitude)]

crs = {'init' : 'epsg:4326'}

gdf_world_copy = GeoDataFrame(world_frame_map_copy, crs=crs, geometry=geometry_copy)
# save the GeoDataFrame

gdf_world_copy.to_file(driver = 'ESRI Shapefile', filename= "CoronaVirus.shp")
world_frame_map=  world_frame_map[world_frame_map['Confirmed_Cases']>0]
world_frame = world_frame[world_frame['Confirmed_Cases']>0]
world_frame_map = world_frame_map.reindex(world_frame.index.repeat(world_frame.Confirmed_Cases))
geometry = [Point(xy) for xy in zip(world_frame_map.Latitude, world_frame_map.Longitude)]

world_frame_map = world_frame_map.drop(['Confirmed_Cases', 'Province', 'Deaths'], axis = 1)

crs = {'init' : 'epsg:4326'}

gdf_world = GeoDataFrame(world_frame_map, crs=crs, geometry=geometry)
gdf_world_indexed = gdf_world.set_index('Country')
plot_dict = gdf_world.Country.value_counts()

plot_dict.head()
#read world map shape file

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# Create a base map

m_2 = folium.Map(location=[25.0376,76.4563], tiles='openstreetmap', zoom_start=2)



# Add a heatmap to the base map

HeatMap(data=world_frame_map[['Latitude', 'Longitude']], radius=10).add_to(m_2)



# Display the map

m_2