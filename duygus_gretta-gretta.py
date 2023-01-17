import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bq_helper import BigQueryHelper

from google.cloud import bigquery

# Set use_legacy_sql to True to use legacy SQL syntax.

job_config = bigquery.QueryJobConfig()

job_config.use_legacy_sql = True
noaa_dataset = BigQueryHelper(

        active_project= "bigquery-public-data", 

        dataset_name = "noaa_gsod")

noaa_dataset.list_tables()
# Let's see what's inside

noaa_dataset.head('stations', num_rows=10)
query = """

    SELECT *

    FROM `bigquery-public-data.noaa_gsod.stations`

    WHERE state = "NJ"

"""

NJ_stations = noaa_dataset.query_to_pandas(query)

NJ_stations[NJ_stations['name'].str.contains('NEWARK')]
# Get all the weather data coming from Newark Intl Airport Station

query = """SELECT year,mo,da,temp,dewp,visib,wdsp,prcp,fog,rain_drizzle,snow_ice_pellets,hail,thunder,tornado_funnel_cloud

    FROM `bigquery-public-data.noaa_gsod.gsod*`

    WHERE stn = '725020'

    """

# Estimate the size

noaa_dataset.estimate_query_size(query)
# take Newark data 

newark_data = noaa_dataset.query_to_pandas(query)

# newark_data.to_csv('noaa_newark_data.csv')

newark_data_sorted = newark_data.sort_values(by=['year','mo','da']).reset_index().drop('index',axis=1)

newark_data_sorted.to_csv('newark_data_sorted.csv', index=False)
query = """

    SELECT *

    FROM `bigquery-public-data.noaa_gsod.gsod*`

    WHERE stn IN ('725020','725030','722950','727410')

    ORDER BY year, mo, da

    """

noaa_dataset.estimate_query_size(query)
all_states = noaa_dataset.query_to_pandas(query)
all_states.to_csv('all_states_data_sorted.csv', index=False)