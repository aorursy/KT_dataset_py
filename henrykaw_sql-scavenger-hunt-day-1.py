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
#Query that find the distinct countries which doesn't
#use ppm as measurement unit to pollution.
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
#Checking the size of query
open_aq.estimate_query_size(query)
# Returning the countries that doesn't use pmm
# Running query if below 10 MB
countries_not_using_pmm = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.01)
#Printing countryies
countries_not_using_pmm.country.tolist()
#Query for the column pollutant and select those with value of 0
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
#Checking the query size
open_aq.estimate_query_size(query)
#Creating pandas DF for pollutants that have a value of 0
pollutant_equal_null = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.01)
#Printing the pollutants and how many times they have a value of 0
pollutant_equal_null.pollutant.value_counts()
import seaborn as sns
import matplotlib.pyplot as plt
#Setting plot size
plt.figure(figsize=(13,5))
#ploting the distribution of pollutants that have value of 0
sns.countplot(pollutant_equal_null.pollutant)