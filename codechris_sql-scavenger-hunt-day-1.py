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

#query to select all the items from the country column where 
# the pollutant column is not pm

query = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            """
# USe the query_to_pandas_safe 
other_unit = open_aq.query_to_pandas_safe(query)
#import libraries
#pycountry for converting the names of the countries
import numpy as np
import pycountry

# filtering the array in order to have each country only once
countries = np.unique(other_unit)
#create a new list to put the countries names in
countries_names = []

#convert the iso 3166 to names
for country in countries :
    for code in pycountry.countries :
        if country == code.alpha_2 :
            countries_names.append(code.name)
            break
        
    
#Alphabet sort the list
countries_names.sort()
print(countries_names)
query = """ SELECT country
            from `bigquery-public-data.openaq.global_air_quality`
            where value = 0
            """
zero = open_aq.query_to_pandas_safe(query)

# filtering the array in order to have each country only once
zero_countries = np.unique(zero)
#create a new list to put the countries names in
zeros_countries_names = []

#convert the iso 3166 to names
for country in zero_countries :
    for code in pycountry.countries :
        if country == code.alpha_2 :
            zeros_countries_names.append(code.name)
            break
        
#Alphabet sort the list
zeros_countries_names.sort()
print(zeros_countries_names)