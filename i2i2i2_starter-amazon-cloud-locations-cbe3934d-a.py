import pandas as pd 



from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame

dfAwsLoc = pd.read_csv('/kaggle/input/aws_cloud_locations.csv')

dfAwsLoc.head(5)
#lat/long points

geometry = [Point(xy) for xy in zip(dfAwsLoc['Long'], dfAwsLoc['Lat'])]

gdf = GeoDataFrame(dfAwsLoc, geometry=geometry)   



#simple map

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(10, 6)), marker='o', color='red', markersize=15);