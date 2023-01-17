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

# query to select Which countries use a unit other than ppm to measure 
# any type of pollution?
# "unit" column != ppm
query_countries = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'"""

countries_unit_no_ppm = open_aq.query_to_pandas_safe(query_countries)

print(countries_unit_no_ppm.country.tolist())

query_pollutants_value_zero = """SELECT DISTINCT pollutant
                   FROM `bigquery-public-data.openaq.global_air_quality`
                   WHERE value = 0
                   """
pollutants_value_zero = open_aq.query_to_pandas_safe(query_pollutants_value_zero)

print(pollutants_value_zero.pollutant.tolist())
countries_unit_no_ppm.to_csv('countriesUnitNoPPM.csv', index = True)
pollutants_value_zero.to_csv('pollutantsValueZero.csv', index = True)