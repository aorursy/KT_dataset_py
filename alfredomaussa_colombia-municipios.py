import pandas as pd

import geopandas as gpd

import numpy as np

# from geopandas import GeoDataFrame

# from shapely.geometry import Point



import folium

from folium import Choropleth

from folium.plugins import HeatMap



data_geo = gpd.read_file("../input/mapa-de-colombia-con-municipios/MunicipiosVeredas19MB.json")

data_geo.head()

data_geo['new']=np.arange(len(data_geo))

print(len(data_geo))
data_geo.index=map(lambda p : str(p),data_geo.index)
# Create a base map

m = folium.Map(location=[4.570868 , -74.297333], tiles='cartodbpositron', zoom_start=5)



# Add a choropleth map to the base map

Choropleth(geo_data=data_geo.geometry.__geo_interface__, 

           data=data_geo.new, 

           key_on="feature.id", 

           fill_color='YlGnBu', 

           legend_name='Major crimi'

          ).add_to(m)



# Display the map

m
data_geo.plot()