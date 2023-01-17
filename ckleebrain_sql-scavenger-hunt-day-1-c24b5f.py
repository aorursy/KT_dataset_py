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
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
#open_aq.list_tables()
#open_aq.head('global_air_quality',10)
# By above two lines, you can look a glance a sample of the table, and can control the number fo the row.
# I was just put 10. it can vary depends on your preference.

#query = """SELECT unit, count(city) as city_count
#            FROM `bigquery-public-data.openaq.global_air_quality`
#            Group by unit
#        """
# cities = open_aq.query_to_pandas_safe(query)
# cities.head()
# According to above query, there are two unit types (ug/m3, ppm) in this table to measure air quality,

query = """SELECT count(distinct country) as cnty_cnt
            FROM `bigquery-public-data.openaq.global_air_quality`
            where unit != 'ppm'
        """
counties = open_aq.query_to_pandas_safe(query)
counties
# therefore, 64 countries use a different unit when they measure their air condition.
query = """SELECT distinct pollutant 
            FROM `bigquery-public-data.openaq.global_air_quality`
            where value = 0
        """
pollution = open_aq.query_to_pandas_safe(query)
pollution
# therefore, seven pollutants are somtimes equvalent to 0 value in various cities.