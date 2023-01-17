# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset

open_aq.head("global_air_quality",num_rows=10)
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
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant!='ppm'
        """
open_aq.estimate_query_size(query)
notppm= open_aq.query_to_pandas_safe(query)
notppm.country.value_counts()
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0
        """
open_aq.estimate_query_size(query)
pollutant = open_aq.query_to_pandas_safe(query)
pollutant.pollutant.value_counts()
query1 = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country='IN'
        """
open_aq.estimate_query_size(query1)
ind=open_aq.query_to_pandas_safe(query1)
ind
