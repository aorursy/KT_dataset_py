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
# Here goes nothing!
# Which countries use a unit other than ppm to measure any type of pollution? 

import bq_helper
air = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")

#What tables have we got?
air.list_tables()

#check the first 5 rows...
air.head('global_air_quality')
#query time!

query1 = """SELECT DISTINCT country 
            FROM `bigquery-public-data.openaq.global_air_quality` 
            WHERE unit != 'ppm'"""

not_ppm = air.query_to_pandas_safe(query1)

#let's check it worked
print(not_ppm)
#explore time!
#this second query holds onto every country entry instead of keeping them unique.
#this means we can plot the results to see how many cities in each of the countries
#are measuring in a different unit to ppm

query2 = """SELECT country 
            FROM `bigquery-public-data.openaq.global_air_quality` 
            WHERE unit != 'ppm'"""

not_ppm_2 = air.query_to_pandas_safe(query2)

#credit to Chitral Puthran for the plotting

import matplotlib.pyplot as pl
import seaborn as sb

pl.figure(figsize = (20, 6))
sb.countplot(not_ppm_2['country'])
#time to count!

num = int(not_ppm.count())

print('There are '+ str(num) + ' countries who do not measure air pollution in ppm!')


# Challenge 2! Which pollutants have a value of exactly 0?
# Let's remind ourselves of the dataset

air.list_tables()


air.table_schema('global_air_quality')

# We can see there's a column labelled 'pollutant' and a column named 'value'. Now to make the query!
query3 = """SELECT pollutant 
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""

ans3 = air.query_to_pandas_safe(query3)

num3 = int(ans3.count())

print('There are '+ str(num3) +' pollutants with the value 0!')