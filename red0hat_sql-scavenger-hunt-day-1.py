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
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Define Dataset with bq_helper
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                      dataset_name = "openaq")
openaq.head("global_air_quality")
# Which cpountries use a unit other than ppm to measure any type of pollution?
# Check what are the units in use, first. and how many rows are in the table.

units_query = """SELECT DISTINCT unit
                 FROM `bigquery-public-data.openaq.global_air_quality` """
units = openaq.query_to_pandas_safe(units_query)

print(units)

count_query = """SELECT count(unit) as row_count 
                 from `bigquery-public-data.openaq.global_air_quality`"""
count = openaq.query_to_pandas_safe(count_query)
print(count)

openaq.table_schema("global_air_quality")
# There are 2 values for the units. Select all countries that do not use 'ppm' as the unit.

query2= """SELECT DISTINCT country 
           FROM `bigquery-public-data.openaq.global_air_quality` 
           WHERE unit != 'ppm'
           ORDER BY country"""

countries_not_ppm = openaq.query_to_pandas_safe(query2)

print(countries_not_ppm)
# Which pollutants have a value of 0.  Possible values are: PM25, PM10, SO2, NO2, O3, CO, BC.

query3= """SELECT DISTINCT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality` 
           WHERE value = 0
           ORDER BY pollutant"""
pollutants_zero = openaq.query_to_pandas_safe(query3)


pollutants_zero