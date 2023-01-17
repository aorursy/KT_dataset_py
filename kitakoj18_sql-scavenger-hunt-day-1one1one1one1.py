import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.table_schema('global_air_quality')
open_aq.head('global_air_quality')
query = """
        SELECT city FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US' 
        """

us_cities = open_aq.query_to_pandas_safe(query)
us_cities.head()
us_cities['city'].value_counts().head()
unit_query = """
             SELECT country 
                 FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != 'ppm'
             """

non_ppm_co = open_aq.query_to_pandas_safe(unit_query)
non_ppm_co['country'].unique()
zero_pollu_query = """
                 SELECT pollutant
                     FROM `bigquery-public-data.openaq.global_air_quality`
                 WHERE value = 0
                 """

zero_pollutants = open_aq.query_to_pandas_safe(zero_pollu_query)
zero_pollutants['pollutant'].unique()
