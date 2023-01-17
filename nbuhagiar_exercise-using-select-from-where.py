# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
query = """SELECT country 
           FROM `bigquery-public-data.openaq.global_air_quality` 
           WHERE pollutant != 'pm25' AND pollutant != 'pm10'"""
non_ppm_countries = open_aq.query_to_pandas_safe(query)
print(non_ppm_countries["country"].unique())
query = """SELECT pollutant 
           FROM `bigquery-public-data.openaq.global_air_quality` 
           WHERE value = 0"""
zero_value_pollutants = open_aq.query_to_pandas_safe(query)
print(zero_value_pollutants["pollutant"].unique())
