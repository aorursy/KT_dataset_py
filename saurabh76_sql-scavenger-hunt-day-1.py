# import package with helper functions 
import bq_helper
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                   dataset_name='openaq')

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head('global_air_quality', num_rows=10)
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city 
from `bigquery-public-data.openaq.global_air_quality`
where country = 'US'
"""
# query = """SELECT city
#             FROM `bigquery-public-data.openaq.global_air_quality`
#             WHERE country = 'US'
#         """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# us_cities.city.value_counts().head()
# Your code goes here :)
query_ctry_not_ppm = """SELECT country 
from `bigquery-public-data.openaq.global_air_quality`
where unit != 'ppm'
"""

# See if the query can be executed safely without hitting the limit 
countries_not_ppm = open_aq.query_to_pandas_safe(query_ctry_not_ppm)
#See some values
countries_not_ppm.country.value_counts()
query_result_zero = """SELECT pollutant 
from `bigquery-public-data.openaq.global_air_quality`
where value = 0.00
"""
pollutant_result_zero = open_aq.query_to_pandas_safe(query_result_zero)
pollutant_result_zero.pollutant.value_counts()