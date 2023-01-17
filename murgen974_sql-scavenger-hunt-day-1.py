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
#import package with helper function
import bq_helper

#Create a helper object 
dataset_air_quality = bq_helper.BigQueryHelper(active_project ="bigquery-public-data",
                                               dataset_name="openaq")

#check list of tables
dataset_air_quality.list_tables()

#print 1st rows
dataset_air_quality.head("global_air_quality")

#country not using ppm
query_unit =""" SELECT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'
"""
country_notusing_ppm = dataset_air_quality.query_to_pandas_safe(query_unit)
country_notusing_ppm.country.value_counts().head()
country_notusing_ppm.country.value_counts()

#Query 2 pollutants with a value of 0

query_pollutantNull = """ SELECT pollutant
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE value = 0
"""
Pollutant_withvalueZero = dataset_air_quality.query_to_pandas_safe(query_pollutantNull)
Pollutant_withvalueZero.pollutant.value_counts()