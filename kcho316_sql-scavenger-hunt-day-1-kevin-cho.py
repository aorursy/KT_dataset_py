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
open_aq.head("global_air_quality")
# Countries that use a unit other than ppm to measure pollution 
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
countries_no_ppm = open_aq.query_to_pandas_safe(query)
# countries that measure a unit other than ppm to measure any type of pollution
countries_no_ppm
# pollutants that have a value of exactly 0 
query = """SELECT pollutant, value, location, timestamp
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant, location, timestamp
        """
zero_pollutants = open_aq.query_to_pandas_safe(query)
zero_pollutants