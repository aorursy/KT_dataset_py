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
query_not_ppm = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country != 'ppm'
        """

open_aq.estimate_query_size(query_not_ppm)
not_ppm = open_aq.query_to_pandas_safe(query_not_ppm)
not_ppm['country'].value_counts().head(10).plot.bar(figsize=(12,6))
not_ppm['country'].unique().size
query_not_ppm_distinct = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country != 'ppm'
        """
open_aq.estimate_query_size(query_not_ppm_distinct)
not_ppm_distinct = open_aq.query_to_pandas_safe(query_not_ppm_distinct)
not_ppm_distinct.count()
query_zero_pollutant = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
zero_pollutant = open_aq.query_to_pandas_safe(query_zero_pollutant)
zero_pollutant['pollutant'].value_counts().plot.bar(figsize=(12,6))