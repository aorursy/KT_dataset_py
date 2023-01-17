import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import our bq_helper package
import bq_helper 

# create a helper object for our bigquery dataset
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm'
            GROUP BY country
        """
open_aq.estimate_query_size(query1)
ppm_countries = open_aq.query_to_pandas_safe(query1, max_gb_scanned=1)
# save our dataframe as a .csv 
ppm_countries.to_csv("ppm_countries.csv")
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
            GROUP BY country"""

# check how big this query will be
open_aq.estimate_query_size(query)
non_ppm_countries = open_aq.query_to_pandas_safe(query, max_gb_scanned=1)
# save our dataframe as a .csv 
non_ppm_countries.to_csv("non_ppm_countries.csv")
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 AND value IS NOT NULL
            GROUP BY pollutant"""

# check how big this query will be
open_aq.estimate_query_size(query2)
zero_pollutants = open_aq.query_to_pandas_safe(query2, max_gb_scanned=1)
# save our dataframe as a .csv 
zero_pollutants.to_csv("zero_pollutants.csv")