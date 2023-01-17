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
import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                    dataset_name = "openaq")
open_aq.list_tables()

open_aq.head("global_air_quality")

no_ppm = """
            Select country
            From `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
            """

country_no_ppm = open_aq.query_to_pandas_safe(no_ppm)

country_no_ppm.head()

country_no_ppm['country'].unique()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (20,8))
sns.countplot(country_no_ppm['country'])

country_no_ppm.to_csv("country_no_ppm.csv")

pp0 = """
            Select pollutant
            From `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            """

poll_pp0 = open_aq.query_to_pandas_safe(pp0)

poll_pp0.head()

poll_pp0['pollutant'].unique()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (10,5))
sns.countplot(poll_pp0['pollutant'])

poll_pp0.to_csv("pollutant0.csv")
