# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head('global_air_quality')
# Your Code Goes Here
query = """SELECT country, pollutant 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'"""
results = open_aq.query_to_pandas_safe(query)
results.country.value_counts().head()
# Your Code Goes Here
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0"""
results = open_aq.query_to_pandas_safe(query)
results.pollutant.value_counts().head()
