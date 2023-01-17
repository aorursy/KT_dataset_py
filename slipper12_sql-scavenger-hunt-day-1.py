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
# query to select all the items from the "country" column where the unit column is not ppm
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
units = open_aq.query_to_pandas_safe(query1)
# import pandas and display countries where the units are not ppm
import pandas as pd
units.country.unique()
# query to select all the items from the "pollutant" column where the "value" column is not equal to 0
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
values = open_aq.query_to_pandas_safe(query2)
# What pollutants are not equal to 0
values.pollutant.value_counts()