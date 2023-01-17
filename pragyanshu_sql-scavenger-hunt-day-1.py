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
#query1 for country column where unit column has value not equal to ppm
query1 = """SELECT country FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'ppm'"""
#to check whether query does not exceed the processing limit of 1GB. It returns the value only if
#it exceeds 1GB
unit_not_ppm = open_aq.query_to_pandas_safe(query1)
#the final output with unique list of countries that have unit not equal to ppm
unit_not_ppm.country.unique()





#query2 for pollutant column where value column has value equal to 0.00
query2 = """SELECT pollutant FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0.00"""
#to check whether query does not exceed the processing limit of 1GB. It returns the value only if
#it exceeds 1GB
pollutant_value_zero = open_aq.query_to_pandas_safe(query2)
#the final output with unique list of pollutants that have value equal to 0.0
pollutant_value_zero.pollutant.unique()



