# import bq_helper package with helper functions 
import bq_helper
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                  dataset_name = "openaq")
# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print information on all the columns in the "global_air_quality" table
# in the OpenAQ dataset
open_aq.table_schema("global_air_quality")
# preview the first couple lines of the "global_air_quality" table
open_aq.head("global_air_quality")
# this query looks in the global_air_quality table in the OpenAQ
# dataset, then gets the country column from every row where 
# the unit column has not "ppm" in it.
query = """SELECT country
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit !='ppm' """
# check how big this query will be
open_aq.estimate_query_size(query)
# as the size is very small, we run the query using query_to_pandas
open_aq.query_to_pandas(query)
# now check how many different values unit column is having in global_air_quality table
query = """SELECT DISTINCT unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           """
# check how big this query will be
open_aq.estimate_query_size(query)
# as the size is very small, we run the query using query_to_pandas
open_aq.query_to_pandas(query)
# the query checks which pollutants have value as Zero
query = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0 """
# check how big this query will be
open_aq.estimate_query_size(query)
# though the size is very less, now use safe mode to run the query
open_aq.query_to_pandas_safe(query)
# now check how many distinct pollutants having the value of Zero
query = """SELECT DISTINCT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0 """
# though the size will be minimal still its good practise to check
open_aq.estimate_query_size(query)
# run in safe mode to avoid more than 1GB scan
open_aq.query_to_pandas_safe(query)
# we have 7 such pollutants
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

