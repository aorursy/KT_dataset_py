# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
open_aq.table_schema("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = "US" """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
query_unit = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
country_unit_not_ppm = open_aq.query_to_pandas_safe(query_unit)
country_unit_not_ppm.country.unique()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 6))
sns.countplot(country_unit_not_ppm['country'])
query_pollutants = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollutants_zero = open_aq.query_to_pandas_safe(query_pollutants)
pollutants_zero.pollutant.unique()
plt.figure(figsize = (20, 6))
sns.countplot(pollutants_zero['pollutant'])