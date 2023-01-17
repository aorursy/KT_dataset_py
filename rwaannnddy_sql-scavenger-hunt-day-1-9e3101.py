# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# import helper 
import bq_helper

# create helper object
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "openaq")

# list all the tables in this data
open_aq.list_tables()
open_aq.head("global_air_quality")
# Making a query that would select the countries that does not use ppm as a pollutant
query = """ SELECT country 
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query)
# This query will check if it is less than 1 gb

country_pollutant = open_aq.query_to_pandas_safe(query)
country_pollutant.head()
country_pollutant.country.value_counts()
query = """ SELECT DISTINCT country, unit 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
            ORDER BY country
"""
open_aq.query_to_pandas_safe(query)
# Find which pollutants have the value of 0
# import bq_helper
import bq_helper

# import helper object
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
query2 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
"""

open_aq.estimate_query_size(query2)
result2 = open_aq.query_to_pandas_safe(query2)
result2.pollutant.value_counts()

open_aq.query_to_pandas_safe(query2)