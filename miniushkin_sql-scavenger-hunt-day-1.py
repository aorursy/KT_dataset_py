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
query_not_ppm = """SELECT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'
                """
coutry_not_ppm = open_aq.query_to_pandas_safe(query_not_ppm)

coutry_not_ppm['country'].unique()
# Which pollutants have a value of exactly 0?

query_zero_pollutant = """SELECT pollutant
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value = 0
                """
coutry_zero_pollutant = open_aq.query_to_pandas_safe(query_zero_pollutant)

coutry_zero_pollutant['pollutant'].unique()
query = """SELECT timestamp, value, unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE country = 'RU' AND city = 'Moscow' AND pollutant = 'co'
                """
co_Moscow = open_aq.query_to_pandas_safe(query)
co_Moscow