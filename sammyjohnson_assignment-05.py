import os

import geopandas

import fiona 

import shapely 

import folium
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world
# retrieve crs info

print(world.crs)



# transform the data into different crs

world_tr = world.to_crs('EPSG:3395')

world.plot()

world_tr.plot()



# create a buffer

world_tr.buffer(5)