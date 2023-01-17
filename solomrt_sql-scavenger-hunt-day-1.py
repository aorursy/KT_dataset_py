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
# Question 1 on units
# (a) List different unit types
query_units = """SELECT unit
FROM `bigquery-public-data.openaq.global_air_quality`"""
df_units= open_aq.query_to_pandas_safe(query_units)
df_units.unit.value_counts()
# (b) Query countries with units different from ppm
query_noppm = """SELECT country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm' """
df_noppm= open_aq.query_to_pandas_safe(query_noppm)
df_noppm.country.value_counts().head()
# Question 2 on null pollutant
query_zero = """SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0 """
df_zero= open_aq.query_to_pandas_safe(query_zero)
df_zero.pollutant.value_counts()
# Extra: query on countries
query_countries = """SELECT country
                     FROM `bigquery-public-data.openaq.global_air_quality` """
countries = open_aq.query_to_pandas_safe(query_countries)
countries.country.value_counts()[:10]