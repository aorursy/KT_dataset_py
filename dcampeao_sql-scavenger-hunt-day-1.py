# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
#My Code - Part 1
# this query looks in the full table in the global air quality
# dataset, then gets the country column from every row where 
# the unit column has a value different from "ppm" in it.
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """
Countries = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
Countries['country'].unique()
#My Code - Part 2
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
Pollutants0 = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
Pollutants0['pollutant'].unique()