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
# question 1
query = """SELECT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
countries_not_ppm = open_aq.query_to_pandas_safe(query)

num_countries_not_ppm = len(countries_not_ppm.country.unique())
print('There are %s countries which measure a pollution statistic in units other than ppm.' %num_countries_not_ppm)
print('The top 10 (by how frequently they appear in the dataset) are:')
countries_not_ppm.country.value_counts().head(10)
# question 2
query = """SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
pollutants_zero = open_aq.query_to_pandas_safe(query)
print('The following pollutants have a value of exactly 0:')
pollutants_zero.pollutant.unique().tolist()