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
# Your code goes here :)
'''
1. Which countries use a unit other than ppm to measure any type of pollution?
(Hint: to get rows where the value *isn't* something, use "!=")

2. Which pollutants have a value of exactly 0?
'''

# query selecting distinct country, unit columns while returning the condition unit NOT 'ppm'
query_2 = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """

unit_test = open_aq.query_to_pandas_safe(query_2)
unit_test
# query selecting distinct polluntant column elements while returning the condition value = 0
query_3 = """SELECT DISTINCT pollutant, value
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE value = 0.00
             ORDER BY pollutant
          """
null_pollutant = open_aq.query_to_pandas_safe(query_3)
null_pollutant
