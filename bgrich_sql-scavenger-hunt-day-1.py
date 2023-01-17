# Import the bq_helper package and pandas
import bq_helper
import pandas as pd

# Create a helper object for the OpenAQ dataset
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", 
                                  dataset_name = "openaq")

# List the tables in the OpenAQ dataset
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT Country
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query) * 1000
countries_no_ppm = open_aq.query_to_pandas_safe(query)
print(countries_no_ppm.head())
print(countries_no_ppm.size)
distinct_countries = countries_no_ppm.Country.unique()
print(distinct_countries)
query_2 = """SELECT location, city, country, pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0
          """
open_aq.estimate_query_size(query_2) * 1000
pollutant_zero = open_aq.query_to_pandas_safe(query_2)
print(pollutant_zero.shape)
print(pollutant_zero.head(n = 10))