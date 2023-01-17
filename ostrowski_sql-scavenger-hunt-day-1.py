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
# Setting QUERYs
QUERY1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
            """
QUERY2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            """
# Solving problem 1
# Finding countries that don't use ppm as standard unit: diff_than_ppm
diff_than_ppm = open_aq.query_to_pandas_safe(QUERY1)
# Print unique values and amount of countries
print('List of countries:\n', diff_than_ppm.country.unique())
print('\n\n{} countries don\'t use ppm as a standard unit.'.format(len(diff_than_ppm.country.unique())))
# Solving problem 2
# Finding pollutants with null value: null_pollutant
null_pollutant = open_aq.query_to_pandas_safe(QUERY2)
# Print unique values and amount of countries
print('List of countries:\n', null_pollutant.pollutant.unique())
print('\n\n{} pollutants have null values in our database.'.format(len(null_pollutant.pollutant.unique())))