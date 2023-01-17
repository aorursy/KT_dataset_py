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

# I use max_db_scanned = 0.5 to limit at 0.5 GB
us_cities = open_aq.query_to_pandas_safe(query, max_gb_scanned=0.5)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Question1: which countries use a unit other ppm to measure any type of pollution?

question1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# I use max_db_scanned = 0.5 to limit at 0.5 GB
no_ppm = open_aq.query_to_pandas_safe(question1, max_gb_scanned=0.5)
print(no_ppm.head())

# Question2: which pollutants have a value of exactly 0?

question2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# I use max_db_scanned = 0.5 to limit at 0.5 GB
value_0 = open_aq.query_to_pandas_safe(question2, max_gb_scanned=0.5)
print(value_0.head())