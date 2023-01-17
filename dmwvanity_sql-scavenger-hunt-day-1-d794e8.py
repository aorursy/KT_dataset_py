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
# Select countries for which the air quality unit of measure is not ppm
query = """SELECT country, unit FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != "ppm" """
not_ppm = open_aq.query_to_pandas_safe(query)
not_ppm.head()
not_ppm.country.value_counts()
# How many countries are there?
query = """SELECT COUNT(DISTINCT country) AS country_count FROM `bigquery-public-data.openaq.global_air_quality` """
country_count = open_aq.query_to_pandas_safe(query)
country_count
# Which pollutants have a value of exactly 0?
query = """SELECT pollutant FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0 """
zero_value = open_aq.query_to_pandas_safe(query)
zero_value.head()
zero_value.pollutant.value_counts()