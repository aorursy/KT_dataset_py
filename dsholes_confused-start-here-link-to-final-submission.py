import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Point
import os
dept_of_interest = "Dept_37-00027"
dept_folder = "../input/data-science-for-good/cpe-data/" + dept_of_interest + "/"

census_data_folder, police_shp_folder, police_csv = os.listdir(dept_folder)

# First we'll look at the Police data
for file in os.listdir(dept_folder+police_shp_folder):
    if ".shp" in file:
        shp_file = file

# Use Geopandas to read the Shapefile
police_shp_gdf = gpd.read_file(dept_folder+police_shp_folder+'/'+shp_file)

# Use Pandas to read the "prepped" CSV, dropping the first row, which is just more headers
police_arrest_df = pd.read_csv(dept_folder+police_csv).iloc[1:].reset_index(drop=True)
police_shp_gdf.head()
police_arrest_df.head()
latlon_exists_index = police_arrest_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index

# Only use subset of data with existing Lat and Lon, to avoid Geocoding addresses or
# "guessing" at the meaning of Y_COORDINATE and Y_COORDINATE.1
police_arrest_df = police_arrest_df.iloc[latlon_exists_index].reset_index(drop=True)
police_arrest_df['LOCATION_LATITUDE'] = (police_arrest_df['LOCATION_LATITUDE']
                                         .astype('float'))
police_arrest_df['LOCATION_LONGITUDE'] = (police_arrest_df['LOCATION_LONGITUDE']
                                         .astype('float'))
# important to check if order in Shapefile is Point(Longitude,Latitude)
police_arrest_df['geometry'] = (police_arrest_df
                                .apply(lambda x: Point(x['LOCATION_LONGITUDE'],
                                                       x['LOCATION_LATITUDE']), 
                                       axis=1))
police_arrest_gdf = gpd.GeoDataFrame(police_arrest_df, geometry='geometry')
police_arrest_gdf.crs = {'init' :'epsg:4326'}
police_shp_gdf.crs = {'init' :'esri:102739'}
police_shp_gdf = police_shp_gdf.to_crs(epsg='4326')
police_shp_gdf.head()
police_arrest_gdf.head()
fig1,ax1 = plt.subplots()
police_shp_gdf.plot(ax=ax1,column='SECTOR')
police_arrest_gdf.plot(ax=ax1,marker='.',color='k',markersize=4)
fig1.set_size_inches(7,7)
for folder in os.listdir(dept_folder+census_data_folder):
    if 'poverty' in folder:
        poverty_folder = folder
poverty_acs_file_meta, poverty_acs_file_ann = os.listdir(dept_folder+
                                                   census_data_folder+'/'+
                                                   poverty_folder)
# Same idea as above, use pandas for CSV's and geopandas for Shapefiles
census_poverty_df = pd.read_csv(dept_folder+
                             census_data_folder+'/'+
                             poverty_folder+'/'+
                             poverty_acs_file_ann)

census_poverty_df = census_poverty_df.iloc[1:].reset_index(drop=True)

# Rename Census Tract ID column in ACS Poverty CSV to align with Census Tract Shapefile
census_poverty_df = census_poverty_df.rename(columns={'GEO.id2':'GEOID'})

census_tracts_gdf = gpd.read_file("../input/cb-2017-48-tract-500k/"+
                                  "cb_2017_48_tract_500k.shp")

# Merge Census Tract GeoDataFrame (from Shapefile) with ACS Poverty DataFrame
# using the 'GEOID', or Census Tract 11-digit numerical ID.
census_merged_gdf = census_tracts_gdf.merge(census_poverty_df, on = 'GEOID')

# Make sure everything is using EPSG:4326
census_merged_gdf = census_merged_gdf.to_crs(epsg='4326')
fig2,ax2 = plt.subplots()
police_shp_gdf.plot(ax=ax2,column='NAME')
police_arrest_gdf.plot(ax=ax2,marker='.',color='k',markersize=5)
census_merged_gdf.plot(ax=ax2,color='#74b9ff',alpha=.4,edgecolor='white')
fig2.set_size_inches(10,10)

