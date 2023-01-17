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
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            ORDER BY country
        """
not_ppm_countries = open_aq.query_to_pandas(query)
countries = not_ppm_countries.country.value_counts()
countries.head()
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (15, 15))
sns.barplot(countries.values, countries.index)
plt.ylabel("Country", fontsize = 15)
plt.xlabel("count", fontsize = 15)
plt.title("Countries don't use PPM as unit", fontsize = 20)
plt.show()
query = """SELECT DISTINCT pollutant, city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY city
        """
no_pollutant_cities = open_aq.query_to_pandas(query)
no_pollutant_cities
zero_pollutant = no_pollutant_cities.pollutant
plt.figure(figsize = (10, 10))
sns.countplot(zero_pollutant)
plt.xlabel("Pollutant", fontsize = 10)
plt.ylabel("Count", fontsize = 10)
plt.show()