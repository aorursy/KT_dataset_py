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
# query to select all the items from the "country" column where the
# "unit" column is not "ppm"
country_ppm_query = """SELECT country
                       FROM `bigquery-public-data.openaq.global_air_quality`
                       WHERE unit != 'ppm'
                    """# Your code goes here :)

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
non_ppm_countries = open_aq.query_to_pandas_safe(country_ppm_query)
# Question 1: Which countries use a unit other than ppm to measure any type of pollution? 
non_ppm_unique_countries = sorted( set( non_ppm_countries.country ) )
print( non_ppm_unique_countries )
# query to select all the items from the "pollutant" column where the
# "value" column is zero (0)
pollutant_zero_query = """SELECT pollutant
                          FROM `bigquery-public-data.openaq.global_air_quality`
                          WHERE value = 0
                       """# Your code goes here :)

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollutant_zeroes = open_aq.query_to_pandas_safe(pollutant_zero_query)
# Question 2: Which pollutants have a value of exactly 0?
zero_pollutants = sorted( set( pollutant_zeroes.pollutant ) )
print( zero_pollutants )