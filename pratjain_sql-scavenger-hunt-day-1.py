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
query1 = "select country from `bigquery-public-data.openaq.global_air_quality` where unit<>'ppm'"
nonppm_countries = open_aq.query_to_pandas_safe(query1)
nonppm_countries.country.unique()
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize = (20, 10))
sns.countplot(nonppm_countries.country)
# Your code goes here :)
query2 = "select pollutant from `bigquery-public-data.openaq.global_air_quality` where value=0"
pollutant_0 = open_aq.query_to_pandas_safe(query2)
pollutant_0.pollutant.unique()
plt.figure(figsize = (20, 10))
sns.countplot(pollutant_0.pollutant)