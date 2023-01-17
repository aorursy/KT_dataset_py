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
# Your code goes here :)
## Import BigQuery Helper
import bq_helper
## Create a helper object
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="openaq")
# What are the tables contained by this dataset?
open_aq.list_tables()

# Print head of the tables above
open_aq.head("global_air_quality")
## First Question
query = """
SELECT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""
## Create DF
df = open_aq.query_to_pandas_safe(query)
## Print countries to answer first question
df.country.unique()
## Second Question
query = """
SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""
## Create DF
df = open_aq.query_to_pandas_safe(query)
## Print pollutants that have value = 0
df.pollutant.unique()
