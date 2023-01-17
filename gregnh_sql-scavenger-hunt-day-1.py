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
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" 
        """
open_aq.estimate_query_size(query)
countries_with_unit_not_ppm = open_aq.query_to_pandas_safe(query)
countries_with_unit_not_ppm.to_csv('countries_with_unit_not_ppm.csv')
countries_with_unit_not_ppm.country.value_counts().head()
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value =0
            """
open_aq.estimate_query_size(query)
pollutant_value_0 = open_aq.query_to_pandas_safe(query)
pollutant_value_0.to_csv('pollutants_with_value_0.csv')
pollutant_value_0.pollutant.value_counts().head()