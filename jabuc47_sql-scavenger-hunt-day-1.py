# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# returns the countries with records and the number of times a record from the country appears in descending order.
countries = """SELECT country,COUNT(country) as Number_of_occurrences
                FROM `bigquery-public-data.openaq.global_air_quality`
                GROUP BY country
                ORDER BY Number_of_occurrences DESC
            """
#print('The size of the query countries: ',end='')
open_aq.estimate_query_size(countries)

print('Returns countries with the number of times a record from that country appears, in descending order.')
open_aq.query_to_pandas_safe(countries)
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
print('What five cities have the most measurements taken there?')
us_cities.city.value_counts().head(5)
print('Table Schema for global_air_quality:')
open_aq.table_schema('global_air_quality')
# Your code goes here :) 

# returns the countries, unit that do not use unit 'ppm'.
N_ppm = """ SELECT DISTINCT country,unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
        """

#print('The size of the query N_ppm: ',end='')
open_aq.estimate_query_size(N_ppm)
print('The dataframe of the query N_ppm, answering which countries use a unit other than "ppm" and their unit:')
open_aq.query_to_pandas_safe(N_ppm)
# returns the pollutants that equal zero, along with location and country
# ordered by the pollutant and country
Pollutants = """ SELECT pollutant,value,location,country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.0
            ORDER BY pollutant,country
        """
#print('The size of the query Pollutants: ',end='')
open_aq.estimate_query_size(Pollutants)

print('The dataframe of the query Pollutants \n- where the pollutant, measured value of the pollutant is "0.00", location and country is returned, ordered by country: ')
open_aq.query_to_pandas_safe(Pollutants)