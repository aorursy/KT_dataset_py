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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Hint: to get rows where the value *isn't* something, use "!="
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# check how big this query will be
open_aq.estimate_query_size(query1)

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
countries_notppm = open_aq.query_to_pandas_safe(query1)

countries_notppm.country.unique()
countries_notppm.country.nunique()
query2 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm'
        """
open_aq.estimate_query_size(query)
countries_ppm = open_aq.query_to_pandas_safe(query2)

countries_ppm.country.unique()
countries_ppm.country.nunique()
import numpy as np
np.intersect1d(countries_ppm.country.unique(),countries_notppm.country.unique())
query3 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
open_aq.estimate_query_size(query3)
countries = open_aq.query_to_pandas_safe(query3)

countries.country.unique()
countries.country.nunique()
query4 = """SELECT unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
open_aq.estimate_query_size(query4)
units = open_aq.query_to_pandas_safe(query4)

units.unit.unique()
query5 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# check how big this query will be
open_aq.estimate_query_size(query5)
pollutants_zero = open_aq.query_to_pandas_safe(query5)
pollutants_zero.pollutant.unique()