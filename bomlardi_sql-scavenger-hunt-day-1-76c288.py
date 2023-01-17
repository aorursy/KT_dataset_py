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

#1 Which countries use a unit other than ppm to measure pollution
units_query = """ SELECT country
                      FROM `bigquery-public-data.openaq.global_air_quality`
                      WHERE unit != 'ppm'
                      GROUP BY country"""
country_units = open_aq.query_to_pandas_safe(units_query, 0.1)
country_units
#apparently a lot
#2 Which pollutants have a value of exactly 0?
pollutant_query = """SELECT pollutant
                     FROM `bigquery-public-data.openaq.global_air_quality`
                     WHERE value = 0
                     GROUP BY pollutant"""
zero_pollute = open_aq.query_to_pandas_safe(pollutant_query, 0.1)
zero_pollute