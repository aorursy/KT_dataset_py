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
us_cities.count()
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
us_cities.head()
# Your code goes here :)
query = """SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# Answer for first question
non_ppm_countries = open_aq.query_to_pandas_safe(query)
non_ppm_countries.head()
non_ppm_countries.count()
# sanity check to see what the countries use
query = """SELECT distinct country, unit
            FROM `bigquery-public-data.openaq.global_air_quality` order by country
        """
country_units = open_aq.query_to_pandas_safe(query)
country_units.head()
# Which countries use ppm as a measure for any of their polutants
query = """SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm'
        """
ppm_countries = open_aq.query_to_pandas_safe(query)
ppm_countries.head()
ppm_countries.count()
# What do we use, and how much
query = """SELECT distinct(unit), count(*)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country='US' group by unit
        """
open_aq.query_to_pandas_safe(query)
# How many contries are there?
query = """SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
         """
open_aq.query_to_pandas_safe(query).count()
# There are some countries which uses both...i want to find it
query = """SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm' and country in (SELECT distinct(country)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm')
        """
country_overlap = open_aq.query_to_pandas_safe(query)
country_overlap.count()
# look at it all it seems like all the countries use something other than ppm
# and only 14 use ppm for any of their metrics
# Second answer
query = """SELECT distinct(pollutant)
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
        """
zero_polute = open_aq.query_to_pandas_safe(query)
zero_polute.head()
zero_polute.count()