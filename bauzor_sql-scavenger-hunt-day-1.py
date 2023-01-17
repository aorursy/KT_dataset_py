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
#importing bigquery
import bq_helper
import pandas as pd

#opening the project folders and storing it in a variable
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")

#tells me what dataset tables are in the folder
open_aq.list_tables()

#returns/prints top couple rows of table
open_aq.head("global_air_quality")
#checks every columns info
open_aq.table_schema("global_air_quality")
#Finds all queries with the following conditions
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """

query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""
#estimates the size of the query
open_aq.estimate_query_size(query1)
open_aq.estimate_query_size(query2)
#returns list of countries not using ppm and removes countires that repeat
countries_not_using_ppm = open_aq.query_to_pandas_safe(query1)
countries_not_using_ppm_undup = pd.DataFrame.drop_duplicates(countries_not_using_ppm, keep="first")
print(countries_not_using_ppm_undup)
#returns list of pollutants that have a value of 0
pollutants_with_value0 = open_aq.query_to_pandas_safe(query2)
undup_pollutants_zero_value = pd.DataFrame.drop_duplicates(pollutants_with_value0, keep="first")
undup_pollutants_zero_value.pollutant.value_counts().head()
print(undup_pollutants_zero_value)