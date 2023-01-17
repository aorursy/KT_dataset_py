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
# Your code goes here : pollutant, unit
open_aq.table_schema('global_air_quality')

#returns a count of the number of cases in which the unit is not ppm
query_1 = """SELECT country, COUNT(*)
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'
                GROUP BY country"""

open_aq.estimate_query_size(query_1)
countries_not_ppm = open_aq.query_to_pandas_safe(query_1)
countries_not_ppm

#returns a count of the number of cases in which the pollutant has value zero
query_2 = """ SELECT pollutant, COUNT(*)
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value = 0
                GROUP BY pollutant
                """

open_aq.estimate_query_size(query_2)
countries_not_ppm = open_aq.query_to_pandas_safe(query_2)
countries_not_ppm