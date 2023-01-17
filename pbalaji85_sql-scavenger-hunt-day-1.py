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
us_cities.head()
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Let's start by importing BigQuery helper
import bq_helper as bqh

## Next, we initiate a big query helper
airQuality_db = bqh.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
## Like Rachel shows above, there is only one table in the airquality database
airQuality_db.list_tables()

airQuality_db.head("global_air_quality")
## Let's get to the challenge quickly
## First, we write the query to determine countries
query1 = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """
## Import the query into a pandas DataFrame
no_ppm_country = airQuality_db.query_to_pandas_safe(query1)
## Count the number of instances of each country.
print(no_ppm_country.country.value_counts().head())

## Let's move on to the second part of the challenge
query2 = """ SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
## import the second query
pollutant = airQuality_db.query_to_pandas_safe(query2)
##Count the number of instances that each pollutant is zero.
print(pollutant.pollutant.value_counts().head())