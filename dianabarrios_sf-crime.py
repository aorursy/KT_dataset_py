# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import folium
sns.set(style="ticks", color_codes=True)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/incidents/Police_Department_Incident_Reports__2018_to_Present.csv')
df.head()
df.shape
df.dtypes
df['Incident Datetime'] = df['Incident Datetime'].astype('datetime64[ns]')
df['Incident Date'] = df['Incident Date'].astype('datetime64[ns]')
df['Incident Time'] = df['Incident Time'].astype('datetime64[ns]')
df['Report Datetime'] = df['Incident Datetime'].astype('datetime64[ns]')

df['Incident Date'] =  pd.to_datetime(df['Incident Date']).dt.date
#we changed time to hour
df['Incident Time'] =  pd.to_datetime(df['Incident Time']).dt.hour

#new column for month
df['Month'] =  pd.to_datetime(df['Incident Datetime']).dt.month
df.head()
df['Incident Year'].value_counts()
df['Incident Day of Week'].value_counts()
df['Incident Time'].value_counts()
df['Month'].value_counts()
df['Incident Category'].value_counts()
chart = sns.catplot(data=df,x="Incident Category", kind="count")
chart.set_xticklabels(rotation=65, horizontalalignment='right')
plt.show()
df['Incident Subcategory'].value_counts()
df.isnull().sum().sort_values(ascending=False)
df['Latitude'].fillna(value=df['Latitude'].mean(), inplace=True)
df['Longitude'].fillna(value=df['Longitude'].mean(), inplace=True)
for col_name in df.columns:
    if df[col_name].dtypes == 'object':
        unique_cat = len(df[col_name].unique())
        print('Variable ''{col_name} tiene {unique_cat} categorías'.format(col_name=col_name, unique_cat=unique_cat))
df['Police District'].value_counts()
chart = sns.catplot(data=df,x="Police District", kind="count")
chart.set_xticklabels(rotation=65, horizontalalignment='right')
plt.show()
df['Analysis Neighborhood'].value_counts()
df['Analysis Neighborhood'].unique()
chart = sns.catplot(data=df,x="Analysis Neighborhood", kind="count")
chart.set_xticklabels(rotation=65, horizontalalignment='right')
plt.show()
df['Supervisor District'].value_counts()
sns.catplot(data=df,x="Supervisor District", kind="count")
plt.show()
df['Latitude'].value_counts()
df['Longitude'].value_counts()
def generateBaseMap(default_location=[37.784560, -122.407337], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map
ex_map = generateBaseMap()
ex_map
from folium.plugins import HeatMap
df_copy = df.sample(frac =.1) 
df_copy['count'] = 1
base_map = generateBaseMap()
HeatMap(data=df_copy[['Latitude', 'Longitude', 'count']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist(), radius=10, max_zoom=13, min_opacity = 0.1, max_val = 50).add_to(base_map)
base_map
base_map.add_child(folium.ClickForMarker(popup='Potential Spot'))
df_hour_list = []
for hour in df_copy['Incident Time'].sort_values().unique():
    df_hour_list.append(df_copy.loc[df_copy['Incident Time'] == hour, ['Latitude', 'Longitude', 'count']].groupby(['Latitude', 'Longitude']).sum().reset_index().values.tolist())
from folium.plugins import HeatMapWithTime
time_map = generateBaseMap(default_zoom_start=11)
HeatMapWithTime(df_hour_list, radius=5, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(time_map)
time_map
layers_map = generateBaseMap()
folium.TileLayer('Stamen Terrain').add_to(layers_map)
folium.TileLayer('Stamen Toner').add_to(layers_map)
folium.TileLayer('Stamen Water Color').add_to(layers_map)
folium.TileLayer('cartodbpositron').add_to(layers_map)
folium.TileLayer('cartodbdark_matter').add_to(layers_map)
folium.LayerControl().add_to(layers_map)
layers_map
import geopandas
sf_geojson=geopandas.read_file('../input/incidents/Analysis_Neighborhoods.geojson')
sf_geojson.head()
sf_geojson.plot()
df_sample = df.sample(frac =.1) 
from folium.plugins import MarkerCluster

c_map = generateBaseMap()

locations = list(zip(df_sample['Latitude'], df_sample['Longitude']))
icons = [folium.Icon(icon="exclamation", prefix="fa") for _ in range(len(locations))]

cluster = MarkerCluster(locations=locations, icons=icons)
c_map.add_child(cluster)
c_map
# calculating total number of incidents per district
crimedata2 = pd.DataFrame(df['Analysis Neighborhood'].value_counts().astype(float))
crimedata2.to_json('crimeagg.json')
crimedata2 = crimedata2.reset_index()
crimedata2.columns = ['Analysis Neighborhood', 'Number']

# creation of the choropleth
map1 = generateBaseMap()

# add tile layers to the map
tiles = ['stamenwatercolor','cartodbpositron','openstreetmap','stamenterrain']
for tile in tiles:
    folium.TileLayer(tile).add_to(map1)


#adding markers in choropleth map 
map1.add_child(cluster)
    
choropleth = folium.Choropleth(geo_data=sf_geojson,
            name='choropleth',
            data=crimedata2, 
              columns = ['Analysis Neighborhood', 'Number'],
              key_on = 'feature.properties.nhood',
              fill_color = 'YlOrRd', 
              fill_opacity = 0.7, 
              line_opacity = 0.2,
              legend_name = 'Number of incidents per neighborhood').add_to(map1)  

# add labels indicating the name of the community
style_function = "font-size: 15px; font-weight: bold"
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(['nhood'], style=style_function, labels=False))

# create a layer control
folium.LayerControl().add_to(map1)

map1