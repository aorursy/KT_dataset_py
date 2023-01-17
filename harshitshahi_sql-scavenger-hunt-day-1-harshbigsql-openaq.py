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

#Question1: Which countries use a unit other than ppm to measure any type of pollution?

countries_not_ppm = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
#Using the dataframe
not_ppm=open_aq.query_to_pandas_safe(countries_not_ppm)

#Countries don't use ppm as a value !
#Show result
not_ppm

#Question2: Which pollutant has value of exactly 0

pollutants_name = """SELECT DISTINCT pollutant
                    FROM `bigquery-public-data.openaq.global_air_quality`
                    WHERE value = 0
                  """
#Using the dataframe
pollutants= open_aq.query_to_pandas_safe(pollutants_name)

#Show result
pollutants

#Ok! Let's try out some Data Visualization now: First Time 
import seaborn as sns
import matplotlib.pyplot as matplt
matplt.figure(figsize = (20, 6))
sns.countplot(not_ppm['country'])

#Let's get unique values. Final Solution:
not_ppm['country'].unique

#Same for getting visual representation of Polltuants 
matplt.figure(figsize=(20,6))
sns.countplot(pollutants['pollutant'])