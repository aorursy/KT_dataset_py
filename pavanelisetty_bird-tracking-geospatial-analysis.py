# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import pandas as pd

#import geopandas as gpd



from shapely.geometry import LineString



#from learntools.core import binder

#binder.bind(globals())

#from learntools.geospatial.ex2 import *
birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv" , parse_dates=['timestamp'])
birds_df.head()
birds_df['tag-local-identifier'].unique() #there are 11 different birds since these are uniquely identified by tag-local-identifier
import geopandas as gpd


birds = gpd.GeoDataFrame(birds_df , geometry = gpd.points_from_xy(birds_df["location-long"] , birds_df["location-lat"]))



birds.crs = {'init':'epsg:4326'}

# Create the GeoDataFrame

birds = gpd.GeoDataFrame(birds_df, geometry=gpd.points_from_xy(birds_df["location-long"], birds_df["location-lat"]))



# Set the CRS to {'init': 'epsg:4326'}

birds.crs = {'init' :'epsg:4326'}
world_path = gpd.datasets.get_path('naturalearth_lowres')

world = gpd.read_file(world_path)

americas = world.loc[world.continent.isin(['North America','South America'])]

americas.head()
ax=americas.plot(figsize=(10,10),edgecolor='black',color='white',linestyle=':')

birds.plot(ax=ax , markersize=10)
# GeoDataFrame showing path for each bird

path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()

path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)

path_gdf.crs = {'init' :'epsg:4326'}



# GeoDataFrame showing starting point for each bird

start_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[0]).reset_index()

start_gdf = gpd.GeoDataFrame(start_df, geometry=start_df.geometry)

start_gdf.crs = {'init' :'epsg:4326'}



# Show first five rows of GeoDataFrame

start_gdf.head()
path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()

path_df
end_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[-1]).reset_index()

end_gdf = gpd.GeoDataFrame(end_df, geometry=end_df.geometry)

end_gdf.crs = {'init' :'epsg:4326'}

end_gdf.head()
ax=americas.plot(figsize=(10,10),color='white',edgecolor='black',linestyle=':')

path_gdf.plot(ax=ax , markersize = 10)



start_gdf.plot(ax=ax , markersize = 15,color='green')



end_gdf.plot(ax=ax , markersize = 20 , color='red')
protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"

protected_areas = gpd.read_file(protected_filepath)

south_america = americas.loc[americas['continent']=='South America']

ax=south_america.plot(figsize=(10,10),color='white',edgecolor='black')

protected_areas.plot(ax=ax , alpha = 0.4 )
protected_areas['REP_AREA'] #total area
protected_areas['REP_M_AREA'].describe() #marine area
P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])

print("South America has {} square kilometers of protected areas.".format(P_Area))
south_america.head()
#Total AREA of southamerica

totalArea = sum(south_america.geometry.to_crs(epsg=3035).area)/10**6

#units in square kilometers

#caluculated by summing up the area of each country and converting in square kilomtrs
#    % south america is protected 

percentage_protected = P_Area/totalArea

print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))
protected_areas[protected_areas['MARINE']!=2]
ax = south_america.plot(figsize=(10,10), color='white', edgecolor='gray')

protected_areas[protected_areas['MARINE']!='2'].plot(ax=ax, alpha=0.4, zorder=1)

birds[birds.geometry.y < 0].plot(ax=ax, color='red', alpha=0.6, markersize=10, zorder=2)