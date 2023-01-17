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
# ----------------------

# open_aq is our BigQueryHelper object.

# First question:
# Which countries use a unit other than ppm to measure any type of pollution? 
# (Hint: to get rows where the value *isn't* something, use "!=")

# First I want to confirm that each country uses only one unit.
query1 = """SELECT country, COUNT(DISTINCT unit)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country
         """

num_GB_scanned_if_ran = open_aq.estimate_query_size(query1)
print(num_GB_scanned_if_ran)


country_units_count = open_aq.query_to_pandas_safe(query1)
import pandas as pd
pd.set_option('display.max_rows', None)
country_units_count
query2 = """SELECT country, unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'
           GROUP BY country, unit
        """

query2a = """SELECT country, unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           GROUP BY country, unit
        """

print(open_aq.estimate_query_size(query2a))
print(open_aq.estimate_query_size(query2))
countries_without_ppm = open_aq.query_to_pandas_safe(query2)
countries_and_their_units = open_aq.query_to_pandas_safe(query2a)
countries_without_ppm.sort_values('country')
countries_and_their_units.sort_values('country')
query3 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0
            GROUP BY pollutant
         """

print(open_aq.estimate_query_size(query3))
pollutants_value_zero = open_aq.query_to_pandas_safe(query3)
pollutants_value_zero