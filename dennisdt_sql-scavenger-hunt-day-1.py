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
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# do the query for countries without ppm units
no_ppm_countries = open_aq.query_to_pandas_safe(query)
# print countries that do not use ppm
print('Countries that do not use ppm: ', no_ppm_countries.country.unique())
# count number of unique country
print('Total number of countries that do not use ppm: ', no_ppm_countries.country.nunique())
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# query for zero value pollutants
zero_pollutant = open_aq.query_to_pandas_safe(query)
print('Pollutants have a value of 0: ', zero_pollutant.pollutant.unique())
print('Total number of pollutants that have a value of 0: ', zero_pollutant.pollutant.nunique())