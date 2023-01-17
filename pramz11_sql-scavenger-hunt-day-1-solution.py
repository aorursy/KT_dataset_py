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
# Day 1 Scavenger Hunt Question 1
# Which countries use a unit other than ppm to measure any type of pollution? 
# (Hint: to get rows where the value *isn't* something, use "!=")


query1 = """ SELECT country
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE pollutant != 'ppm'
         """

# check query size before executing 

open_aq.estimate_query_size(query1)

# Using query_to_pandas_safe returns a result only if its less than 1GB by default
nonppm_countries = open_aq.query_to_pandas(query1)

#  Five countries that have the most non-ppm measurements
nonppm_countries.country.value_counts().head()


# Day 1 Scavenger Hunt Question 2
# Which pollutants have a value of exactly 0?

query2 = """ SELECT pollutant
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE value = 0
         """

# check the size of the query before executing it 
open_aq.estimate_query_size(query2)

# The size of the query is around 0.2 mb 

# Using query_to_pandas_safe returns a result only if its less than 1GB by default
zero_pollutants = open_aq.query_to_pandas_safe(query2)

# Top five pollutants with zero pollutant value
zero_pollutants.pollutant.value_counts().head()