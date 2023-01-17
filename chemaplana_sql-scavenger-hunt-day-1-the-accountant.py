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
query_ppm = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query_ppm)
df_ppm = open_aq.query_to_pandas_safe(query_ppm)
print (df_ppm['country'].unique())
print ('Number of countries not using ppm %d.' %len(df_ppm['country'].unique()))
# Your code goes here :)
query_ppm2 = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
open_aq.estimate_query_size(query_ppm2)
df_ppm2 = open_aq.query_to_pandas_safe(query_ppm2)
print (df_ppm2['unit'].unique())
# Your code goes here :)
query_zero = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
open_aq.estimate_query_size(query_zero)
df_zero = open_aq.query_to_pandas_safe(query_zero)
print (df_zero['pollutant'].unique())
print ('Number of pollutant equal to zero %d.' %len(df_zero['pollutant'].unique()))
print (df_zero.describe())