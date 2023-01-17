# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
query = """select country
           from `bigquery-public-data.openaq.global_air_quality`
           where unit != 'pm'"""
countries_no_ppm = open_aq.query_to_pandas_safe(query=query)
countries_no_ppm.country.unique()
query = """select pollutant
           from `bigquery-public-data.openaq.global_air_quality`
           where value = 0.0"""
open_aq.head("global_air_quality")
open_aq.estimate_query_size(query)
pollutants_zero = open_aq.query_to_pandas_safe(query=query)
pollutants_zero.pollutant.unique()