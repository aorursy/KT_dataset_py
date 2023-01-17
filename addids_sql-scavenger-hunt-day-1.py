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
#Q1: Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value isn't something, use "!=")
qry_ppm = """SELECT country FROM `bigquery-public-data.openaq.global_air_quality` 
WHERE unit != 'ppm'"""

#check the estimated query size
print("Estimated size for PPM Query:", open_aq.estimate_query_size(qry_ppm))

#load qry_ppm result into pandas df "country_ppm"
country_ppm = open_aq.query_to_pandas_safe(qry_ppm)

#print all countries from query (avoid double listing)
print ("Countries with unit different to ppm:", country_ppm.country.unique())

#Q2: Which pollutants have a value of exactly 0?
qry_pol = """SELECT pollutant FROM `bigquery-public-data.openaq.global_air_quality` 
WHERE value = 0"""

#check the estimated query size
print("Estimated size for pollutant query:", open_aq.estimate_query_size(qry_pol))

#load qry_ppm result into pandas df "country_ppm"
pollutant_value = open_aq.query_to_pandas_safe(qry_pol)

#print all pollutant with value 0
print ("Pollutant with value 0: ", pollutant_value.pollutant.unique())
