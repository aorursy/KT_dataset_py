# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# Question 1: Which countries use a unit other than ppm to measure any type of pollution? 
# (Hint: to get rows where the value isn't something, use "!=")
# Query for countries with unit that does not include ppm
query_for_countries = """SELECT DISTINCT country, unit
                         FROM `bigquery-public-data.openaq.global_air_quality`
                         WHERE unit NOT IN("ppm")
                        """
# Creating data frame for countries(filtered) where the query_to_pandas_safe will only return a result
# if it's less than one gigabyte (by default)
countries_with_no_ppm = open_aq.query_to_pandas_safe(query_for_countries)
print(countries_with_no_ppm)
# Question 2 -> Which pollutant have a value of exactly 0?
# Query for distinct pollutants with value of 0 
query_for_pollutants = """SELECT DISTINCT pollutant, value
                         FROM `bigquery-public-data.openaq.global_air_quality`
                         WHERE value = 0
                        """
# Creating data frame for pollutant where the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollutants = open_aq.query_to_pandas_safe(query_for_pollutants)
print(pollutants)