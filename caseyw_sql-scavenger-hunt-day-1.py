# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality",10)
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
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
# open_aq.list_tables()
open_aq.head("global_air_quality",6)

# Just to show what the text format for units will look like."

unit_query = """SELECT Distinct unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
units = open_aq.query_to_pandas_safe(unit_query)
units
my_query = """SELECT Distinct city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
        """
noppm_cities = open_aq.query_to_pandas_safe(my_query)
noppm_cities.head(10)
noppm_cities.city.value_counts()