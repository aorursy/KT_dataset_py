import os
import geopandas
import fiona
import shapely
import folium
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
world
# retrieve crs info
print(world.crs)

# transform data into a different crs
world_tr = world.to_crs('EPSG:3395')
world_tr.plot()
world.plot()

# create a buffer
#world_buf = world.buffer(1)
world_tr.buffer(5)


