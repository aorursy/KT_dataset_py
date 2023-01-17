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
# Counties not using PPM
query_ppm = """ SELECT DISTINCT Country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE Unit != 'ppm'
"""

countries_ppm = open_aq.query_to_pandas_safe(query_ppm)
print(countries_ppm.to_string())

# Pollutants = 0
query_zero = """ SELECT DISTINCT Pollutant
                 FROM `bigquery-public-data.openaq.global_air_quality`
                 WHERE Value = 0
"""

pollutants_zero = open_aq.query_to_pandas_safe(query_zero)
print(pollutants_zero.to_string())