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
# select countries that use other than 'ppm' to measure any type of pollution
# returning row count by country
query = """SELECT country,count(country)count
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            GROUP BY country
            ORDER BY 2 desc
        """
countries_no_ppm = open_aq.query_to_pandas_safe(query)
countries_no_ppm
# Your code goes here :)
# select pollutants which values is equal '0'
# returning row count by pollutant
query = """SELECT pollutant,
                  count(pollutant)count_all,
                  count(case when value=0 then pollutant else null end)count_iqual_0
            FROM `bigquery-public-data.openaq.global_air_quality`
            --WHERE value = 0
            GROUP BY pollutant
            ORDER BY 2 desc
        """
pollutant = open_aq.query_to_pandas_safe(query)
pollutant