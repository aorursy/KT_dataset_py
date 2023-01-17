import warnings
warnings.filterwarnings('ignore')
from google.cloud import bigquery
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (16, 20)
import pandas_profiling
import cartopy.crs as ccrs
import pandas as pd
# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper
# Use  bq_helper to create a BigQueryHelper object
noaa_goes = BigQueryHelper(active_project= "bigquery-public-data", 
                              dataset_name= "noaa_goes16")
# Get a list of all tables
noaa_goes.list_tables()
noaa_goes.table_schema('abi_l1b_radiance')
%%time 
noaa_goes.head("abi_l1b_radiance", num_rows=20)
query = """ SELECT dataset_name, platform_id, orbital_slot, timeline_id,
       scene_id, band_id, time_coverage_start, time_coverage_end,
       date_created, geospatial_westbound_longitude,
       geospatial_northbound_latitude, geospatial_eastbound_longitude,
       geospatial_southbound_latitude, nominal_satellite_subpoint_lon,
       valid_pixel_count, missing_pixel_count, saturated_pixel_count,
       undersaturated_pixel_count, min_radiance_value_of_valid_pixels,
       max_radiance_value_of_valid_pixels,
       mean_radiance_value_of_valid_pixels,
       std_dev_radiance_value_of_valid_pixels,
       percent_uncorrectable_l0_errors, total_size, base_url
       FROM `bigquery-public-data.noaa_goes16.abi_l1b_radiance` 
       WHERE scene_id='Mesoscale' """
# Get the estimated query size in GB
noaa_goes.estimate_query_size(query)
query2 = """ SELECT  dataset_name, base_url
              FROM `bigquery-public-data.noaa_goes16.abi_l1b_radiance`
              
          """
noaa_goes.estimate_query_size(query2)
