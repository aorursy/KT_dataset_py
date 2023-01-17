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
# import package with helper functions 
import bq_helper
import pandas as pd
import numpy as np
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# print all the tables in this dataset (there's only one!)
#open_aq.list_tables()
# question 1 query
query_not_ppm = """SELECT distinct country,unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            where unit <> 'ppm'
            order by country
            """
# cost of query
print("Query size estimate:")
print(open_aq.estimate_query_size(query_not_ppm))
# execute query 
cnty_not_ppm = open_aq.query_to_pandas_safe(query_not_ppm)

print("Countries that don't use ppm for measuring all pollutants.")
print(cnty_not_ppm)
# question 2: Query
query_zero_pollu_read = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            where value =0.0
            """
# size estimate for running above query:
print("Query size estimate:")
print(open_aq.estimate_query_size(query_zero_pollu_read))
# execute query for question 2:

zero_pollu = open_aq.query_to_pandas_safe(query_zero_pollu_read)

print("Pollutants with at least one reading of 0.")
print(zero_pollu)


# question 2a: Which pollutants have a value of exactly 0?
# This might be a better question:  Which pollutants in what countries have a value of zero?
query_zero_cty_pollu_read = """SELECT country, pollutant, value_total 
            FROM (SELECT country,pollutant, sum(value) as value_total
            FROM `bigquery-public-data.openaq.global_air_quality`
            group by country,pollutant)
            where value_total = 0
            order by country, pollutant
            """
# size estimate for running above query:
print("Query size estimate:")
print(open_aq.estimate_query_size(query_zero_cty_pollu_read))

zero_cty_pollu_tbl = open_aq.query_to_pandas_safe(query_zero_cty_pollu_read)

print("Pollutants per country where all pollutant values are zero.  ")
print(zero_cty_pollu_tbl)


# size estimates for running queries
#open_aq.estimate_query_size(query_not_ppm)
#open_aq.estimate_query_size(query_zero_pollu_read)
