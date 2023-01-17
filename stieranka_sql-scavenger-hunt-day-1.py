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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

open_aq.head('global_air_quality')
query_country="""SELECT  country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """
countries = open_aq.query_to_pandas_safe(query_country)

query_country_no_ppm="""SELECT  country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """

coutries=open_aq.query_to_pandas_safe(query_country_no_ppm)

# List of countries with different type of measure

print('Which countries use a unit other than ppm to measure any type of pollution?')
for country in coutries.country.unique():
    print (country)
# The list of countries with zero value.
# Which pollutants have a value of exactly 0?
query_zero_pollutants = """ SELECT  pollutant
                            FROM `bigquery-public-data.openaq.global_air_quality`
                            WHERE value = 0 """
pollutant=open_aq.query_to_pandas_safe(query_zero_pollutants)

print('Which pollutants have zero value?')
for pollutant in pollutant.pollutant.unique():
    print (pollutant)
print('Suprisingly, not that much.')