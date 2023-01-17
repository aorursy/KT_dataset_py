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
# Which countries use a unit other than ppm to measure any type of pollution?
querySF1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# How much data will I get back if I run this query? (units are in GB)
open_aq.estimate_query_size(querySF1)
# Create a dataframe of returned rows
countries_not_ppm = open_aq.query_to_pandas_safe(querySF1)
# Number of locations grouped by country.
countries_not_ppm.country.value_counts()
# Which pollutants have a value of exactly 0?
querySF2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# How much data will I get back if I run this query? (units are in GB)
open_aq.estimate_query_size(querySF1)
# Create a df for returned rows
zero_pollutants = open_aq.query_to_pandas_safe(querySF2)
# Number of zero values grouped by type of pollutant.
zero_pollutants.pollutant.value_counts()