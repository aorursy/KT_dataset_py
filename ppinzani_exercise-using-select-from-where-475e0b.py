# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
# Your Code Goes Here
query = """SELECT country
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != "ppm" """

countries_no_ppm = open_aq.query_to_pandas_safe(query)
countries_no_ppm.country.value_counts()
# Your Code Goes Here
query = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0 """

pollutant_zero = open_aq.query_to_pandas_safe(query)
pollutant_zero.pollutant.value_counts()