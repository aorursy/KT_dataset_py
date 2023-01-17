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
# Direct query to get each occurance of a country where a unit other than ppm is used
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
        """
open_aq.estimate_query_size(query1)

query2 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
        """
open_aq.estimate_query_size(query2)
query3 = """SELECT country, count(*) as cnt
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
            GROUP BY country 
        """
open_aq.estimate_query_size(query3)
query4 = """WITH country_units as (
                SELECT DISTINCT country, unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                )
            SELECT country 
            FROM country_units 
            WHERE country NOT IN (SELECT DISTINCT country FROM country_units WHERE unit = 'ppm')
        """
open_aq.estimate_query_size(query4)
query5 = """WITH country_units as (
                SELECT DISTINCT country, unit
                FROM `bigquery-public-data.openaq.global_air_quality`
                )
            SELECT cu1.country 
            FROM country_units cu1 
            LEFT JOIN country_units cu2 ON cu1.country = cu2.country AND cu2.unit = 'ppm'
            WHERE cu1.country IS NULL
        """
print("query5: " + str(open_aq.estimate_query_size(query5)))

query6 = """ SELECT cu1.country 
            FROM `bigquery-public-data.openaq.global_air_quality` cu1 
            INNER JOIN `bigquery-public-data.openaq.global_air_quality` cu2 
            ON cu1.country = cu2.country AND cu2.unit = 'ppm'
            WHERE cu1.country IS NULL
        """
print("query6: " + str(open_aq.estimate_query_size(query6)))
# I chose to use query3 to get a table of the countries that uses a unit other than ppm 
countries_non_ppm = open_aq.query_to_pandas_safe(query3)

countries_non_ppm.head()
countries_non_ppm.sort_values(['cnt', 'country'], ascending=[0, 1]).head()
countries_only = countries_non_ppm[['country']]
countries_only.head()
# Which pollutants have a value of exactly 0?
query7 = """ SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
print("query7: " + str(open_aq.estimate_query_size(query7)))
      
# of just the list of pollutants that have a value of 0.00 and number of times it occurs
query8 = """ SELECT pollutant, count(*) as cnt
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00 
            GROUP BY pollutant
        """
      
print("query8: " + str(open_aq.estimate_query_size(query8)))
pollutant_zero_value = open_aq.query_to_pandas_safe(query8)

pollutant_zero_value.head()
pollutant_zero_value.sort_values(['cnt', 'pollutant'], ascending=[0, 1]).head()
pollutants_only = pollutant_zero_value[['pollutant']]

pollutants_only.head()