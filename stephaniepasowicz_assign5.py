import os
import geopandas
import fiona
import shapely
import folium
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world
#check crs
world.crs

# transform
world_tr = world.to_crs('EPSG:3395')
world_tr.plot()

#create buffer
#world_buff = world.buffer(1)
world.buffer = world_tr.buffer(0.001)