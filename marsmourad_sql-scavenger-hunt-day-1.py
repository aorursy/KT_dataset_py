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
# Here is the query to select countries which use a unit other thant ppm
query = """ SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE LOWER(unit) != 'ppm'
            ORDER BY country
        """

non_ppm_countries = open_aq.query_to_pandas_safe(query)

# display query results
non_ppm_countries
# query to select countries having pollutants SUM (value) equal to 0
query3 = """   SELECT country, pollutant, SUM(value) AS Sum_Poll
                       FROM `bigquery-public-data.openaq.global_air_quality`
                       GROUP BY country, pollutant
                       HAVING Sum_Poll = 0
                       ORDER BY country
                """

zero_pol_readings2 = open_aq.query_to_pandas_safe(query3)

# display query results
zero_pol_readings2