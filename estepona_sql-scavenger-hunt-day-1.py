# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                   dataset_name = 'openaq')
# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head('global_air_quality')
open_aq.table_schema('global_air_quality')
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """
# this query costs 0.31MB
open_aq.estimate_query_size(query) * 1024
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
type(us_cities)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Your code goes here :)
query = """
        SELECT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        """

open_aq.estimate_query_size(query) * 1000
non_ppm_countries = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)
non_ppm_countries.shape
non_ppm_countries.country.unique()
# let's see what countries actually use 'ppm'
query = """
        SELECT country, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit = 'ppm'
        """

open_aq.estimate_query_size(query) * 1000
ppm_countries = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)

ppm_countries.head()
ppm_countries.country.unique()
query = """
        SELECT pollutant, value
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """

open_aq.estimate_query_size(query) * 1000
pollu_value0 = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.1)

pollu_value0.head()
pollu_value0.shape
pollu_value0.pollutant.unique()