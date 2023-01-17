# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


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

#Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value isn't something, use "!=")
query_country="""SELECT  country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' """
coutries=open_aq.query_to_pandas_safe(query_country)
print('Which countries use a unit other than ppm to measure any type of pollution?')
for country in coutries.country.unique():
    print (country)


#Which pollutants have a value of exactly 0?
query_pollutant="""SELECT  pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
pollutant=open_aq.query_to_pandas_safe(query_pollutant)
print('Which pollutants have a value of exactly 0?')
for pollutant in pollutant.pollutant.unique():
    print (pollutant)
    

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.