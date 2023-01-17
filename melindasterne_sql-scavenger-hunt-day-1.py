# import packages and helper package
import numpy as np 
import pandas as pd 
import bq_helper

# create a helper object for this dataset
Open = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
Open.list_tables()

#print the first few rows of the global_air_quality table
Open.head('global_air_quality')

#question 1
# query to select all the items from the "country" column where the
# "unit" column != "ppm"
ppmQuery = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """
ppmCountries=Open.query_to_pandas_safe(ppmQuery)
print('Which countries use a unit other than ppm to measure any type of pollution?')
nation = [country for country in ppmCountries.country.unique()]
print (nation)

#question 2    
#which pollutants have a value of exactly 0?

pollutant0 = """SELECT pollutant
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value = 0 """
value0 = Open.query_to_pandas_safe(pollutant0)
print('Which Pollutants have a value of exatly 0?')
xXx = [pollutant for pollutant in value0.pollutant.unique()]
print(xXx)