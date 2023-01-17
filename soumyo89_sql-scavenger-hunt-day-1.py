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

#Solution 1

query1 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# check how big this query will be
open_aq.estimate_query_size(query1)
countries_not_ppm = open_aq.query_to_pandas_safe(query1)
countries_not_ppm.head()
# save our dataframe as a .csv 
countries_not_ppm.to_csv("countries_not_ppm.csv")
# Solution 2
query2 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

# check how big this query will be
open_aq.estimate_query_size(query2)
pollutant_zero = open_aq.query_to_pandas_safe(query2)

# save our dataframe as a .csv 
pollutant_zero.to_csv("pollutant_zero.csv")