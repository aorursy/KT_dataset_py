#install keplergl
!pip install keplergl

#install nodejs
!conda install -y -c conda-forge/label/cf202003 nodejs

#install kepler lab extension
!jupyter labextension install @jupyter-widgets/jupyterlab-manager keplergl-jupyter

#install rtre
!pip install Rtree
import pandas as pd
from keplergl import KeplerGl
import geopandas as gpd
from shapely import wkt

import os
print(os.listdir("../input"))
#enable CSV driver
import fiona as fi
fi.drvsupport.supported_drivers['CSV'] = 'rw'

#load complaint data
complaint_data = gpd.read_file("../input/nypd-2019-complaint-data/NYPD_Complaint_Data_2019_1.csv")
complaint_data.head()
# Create a gepdataframe
complaint_data_gdf = gpd.GeoDataFrame(complaint_data, geometry=gpd.points_from_xy(complaint_data.Longitude.astype('float32'), complaint_data.Latitude.astype('float32')))
complaint_data_gdf.head()
#Create a basemap 
complaint_data_map = KeplerGl(height=600, width=800)

# Add data to map
complaint_data_map.add_data(data=complaint_data_gdf, name="crimes")
complaint_data_map
#load neighbourhood data
#from https://data.beta.nyc/dataset/pediacities-nyc-neighborhoods
#reference https://towardsdatascience.com/how-to-easily-join-data-by-location-in-python-spatial-join-197490ff3544
neighborhood_data = gpd.read_file("../input/nyneighbourhoods/NewyorkNeighbourhoods.json")
neighborhood_data.head()
neighborhood_data_map = KeplerGl(height=600, width=800)
# Add neighbourhood data to Kepler
neighborhood_data_map.add_data(data=neighborhood_data, name="”Neighborhoods”")
neighborhood_data_map
#merge neighbourhood and crime data
def count_incidents_neighborhood(data, neighb):
 # spatial join and group by to get count of incidents in each neighbourhood 
 joined = gpd.sjoin(data, neighb, op="within")
 grouped = joined.groupby('neighborhood').size()
 df = grouped.to_frame().reset_index()
 df.columns = ['neighborhood', 'count']
 merged = neighb.merge(df, on='neighborhood', how='outer')
 merged['count'].fillna(0,inplace=True)
 merged['count'] = merged['count'].astype(int)
 return merged

merged_data = count_incidents_neighborhood(complaint_data_gdf, neighborhood_data)
merged_map = KeplerGl(height=600, width=800)
# Add merged data to Kepler
merged_map.add_data(data=merged_data, name="NeighborhoodCrimes")
merged_map
building_data = gpd.read_file("../input/sanfransisco-building-data/Building_Footprints_2.json")
building_data.head()
building_data_map = KeplerGl(height=600, width=800)
# Add data to Kepler
building_data_map.add_data(data=building_data[:3000], name="Buildings")
building_data_map