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
query_first = """ SELECT country, count(DISTINCT(unit)) as num_measurments
                 FROM `bigquery-public-data.openaq.global_air_quality`
                 GROUP BY country HAVING num_measurments > 1 """

oaq_to_file = open_aq.query_to_pandas_safe(query_first)

oaq_to_file.to_csv("non-ppm measurments.csv")
query_sec = """SELECT pollutant
                       FROM `bigquery-public-data.openaq.global_air_quality`
                       WHERE value = 0 """

pol_to_file = open_aq.query_to_pandas_safe(query_sec)

pol_to_file.to_csv("zero value pollutant.csv")
