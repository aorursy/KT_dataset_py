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
other_unit_query = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
       """
other_unit = open_aq.query_to_pandas_safe(other_unit_query)
open_aq.query_to_pandas_safe(other_unit_query, max_gb_scanned=0.1)
poll0_query = """SELECT location, pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
       """
poll0 = open_aq.query_to_pandas_safe(poll0_query)
open_aq.query_to_pandas_safe(poll0_query, max_gb_scanned=0.1)