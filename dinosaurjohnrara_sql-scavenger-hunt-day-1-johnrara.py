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
#countries that use unit other ppm 
query1 = """SELECT DISTINCT country, unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != "ppm" 
        """
#check how big this query will be
open_aq.estimate_query_size(query1)
#query returns a datafram only if query is smaller than 0.1GB
non_ppm_unit_countries = open_aq.query_to_pandas_safe(query1)
non_ppm_unit_countries
#which pollutants have a value of exactly 0 
query2 = """SELECT DISTINCT pollutant, value
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value = 0
        """
#check how big this query will be
open_aq.estimate_query_size(query2)
#query returns a datafram only if query is smaller than 0.1GB
pollutant_value_zero = open_aq.query_to_pandas_safe(query2)
pollutant_value_zero