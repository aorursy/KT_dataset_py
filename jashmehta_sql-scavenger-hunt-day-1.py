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
query = """SELECT *
            FROM `bigquery-public-data.openaq.global_air_quality`
    
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
# query to select all the items from the "city" column where the
# "country" column is "us"
query1 = """SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
query2 = """ SELECT distinct(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
countries = open_aq.query_to_pandas_safe(query1)
pollutant = open_aq.query_to_pandas_safe(query2)
countries
pollutant