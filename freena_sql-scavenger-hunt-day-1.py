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
query1 = """SELECT country,pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """
result1 = open_aq.query_to_pandas_safe(query1)
result1.tail(10)
query2 = """SELECT country,pollutant,value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
result2 = open_aq.query_to_pandas_safe(query2)
result2.tail(10)
query3 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
result3 = open_aq.query_to_pandas_safe(query3)
result3
query4 = """SELECT city, pollutant, AVG(value) AS average_value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' AND pollutant = 'pm25'
            GROUP BY city,pollutant
            ORDER BY average_value DESC
        """
result4 = open_aq.query_to_pandas_safe(query4)
result4.head(10)