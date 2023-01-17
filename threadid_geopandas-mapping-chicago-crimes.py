# local Operating System
import os

# Visualisation
import geopandas
from geopandas import GeoDataFrame

# Dataframe
import pandas as pd
from shapely.geometry import Point, Polygon

# SQL - PostgreSQL
import bq_helper
from bq_helper import BigQueryHelper
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="chicago_crime") 
print(os.listdir("../input")) 
chicri_geo = geopandas.read_file('../input/geo_export_33ca7ae0-c469-46ed-84da-cc7587ccbfe6.shp')

city_sql_loc = """ 
WITH cr AS
(
    SELECT 
    latitude
    , longitude 
    , primary_type
    , community_area
    FROM  `bigquery-public-data.chicago_crime.crime`
    WHERE year = 2012
    /*WHERE date = '2012-01-15'*/
)
SELECT 
cr.latitude
, cr.longitude
, cr.primary_type
, cr.community_area
FROM cr
ORDER BY  cr.latitude, cr.longitude
"""
chicri_loc_df = chicago_crime.query_to_pandas(city_sql_loc)
chicri_loc_df = chicri_loc_df.dropna(inplace=False)  # Remove all nan entries. 
chicri_loc_df = chicri_loc_df.drop(chicri_loc_df[(chicri_loc_df.latitude < 41.0)].index) #Remove bad values in Long/Lat 
chicri_loc_df['community_area'] = chicri_loc_df['community_area'].astype(int)
chicri_geometry = [Point(xy) for xy in zip(chicri_loc_df.longitude, chicri_loc_df.latitude)]
chicri_crs = {'type': 'EPSG', 'properties': {'code': 102671}}
chicri_points = GeoDataFrame(chicri_loc_df, crs=chicri_crs, geometry=chicri_geometry)
chicri_points.head(5)
chicri_points_map = chicri_points.plot(figsize=(30,30), markersize=5) 
chicri_points_map.set_axis_off()
chicri_map = chicri_geo.plot(figsize=(30,30), edgecolor='k', cmap='nipy_spectral', alpha=0.5, linewidth=2) 
chicri_geo.apply(lambda x: chicri_map.annotate(s=x.community, xy=x.geometry.centroid.coords[0], ha='center', size=16),axis=1);
chicri_map.set_axis_off()
chicri_map = chicri_geo.plot(figsize=(25,25), edgecolor='k', facecolor='b', alpha=0.25, linewidth=2) 
chicri_geo.apply(lambda x: chicri_map.annotate(s=x.community, xy=x.geometry.centroid.coords[0], ha='center', size=16),axis=1);
chicri_points.plot(figsize=(25,25),ax=chicri_map, markersize=5, color='r', alpha=0.25)
chicri_map.set_axis_off()
chicri_vc_points = chicri_points.loc[chicri_points['primary_type'].isin(['ASSAULT'
                                                                , 'BATTERY'
                                                                , 'CRIM SEXUAL ASSAULT'
                                                                , 'HOMICIDE'
                                                                , 'HUMAN TRAFFICKING'
                                                                , 'INTIMIDATION'
                                                                , 'KIDNAPPING'])]

chicri_vc_points_map = chicri_vc_points.plot(figsize=(30,30), markersize=5) 
chicri_vc_points_map.set_axis_off()
chicri_vc_map = chicri_geo.plot(figsize=(25,25), edgecolor='k', facecolor='b', alpha=0.25, linewidth=2) 
chicri_geo.apply(lambda x: chicri_vc_map.annotate(s=x.community, xy=x.geometry.centroid.coords[0], ha='center', size=16),axis=1);
chicri_vc_points.plot(ax=chicri_vc_map, markersize=5, color='r', legend=True)
chicri_vc_map.set_axis_off()