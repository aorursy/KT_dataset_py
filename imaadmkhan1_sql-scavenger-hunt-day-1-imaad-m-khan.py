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
            WHERE country = 'IN'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
in_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
in_cities.city.value_counts().head()
query_0 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            """
# Your code goes here :)
query_1 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
df_0 = open_aq.query_to_pandas_safe(query_0)
print("The countries which do not use ppm as their metric are:")
df_0.country.unique()
df = open_aq.query_to_pandas(query_1)
print("The pollutants with a value of 0 are:")
df.pollutant.unique()