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
# get the table schema so we can find the colunm name turns out to be "unit" and Value...
open_aq.table_schema("global_air_quality")
# Your code goes here :)

# build not eq ppm query
notppmQuery = """SELECT city,unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# Run the query
not_ppm = open_aq.query_to_pandas_safe(notppmQuery)
# validate results
not_ppm.unit.value_counts().head()
# build pollutants values = 0
pollZeroQuery = """SELECT city,value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# Run the Query
poll_zero = open_aq.query_to_pandas_safe(pollZeroQuery)
# validate results
poll_zero.value.value_counts().head()