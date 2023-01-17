# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality", num_rows=10)
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
open_aq.estimate_query_size(query)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
# Scavenger hunt
# 1
# Which countries use a unit other than ppm to measure any type of pollution? 
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query1)
countries1 = open_aq.query_to_pandas_safe(query1)
countries1.country.value_counts().head(10)

# 2
#  Which pollutants have a value of exactly 0?
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value= 0.00
        """
open_aq.estimate_query_size(query2)
pollutant2 = open_aq.query_to_pandas_safe(query2)
pollutant2.pollutant.value_counts().head(10)

