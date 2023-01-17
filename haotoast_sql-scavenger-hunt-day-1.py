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
# Countries use a unit other than PPM
query_non_ppm = """SELECT country from `bigquery-public-data.openaq.global_air_quality`
                    WHERE unit != 'ppm'
                """
non_ppm_countries = open_aq.query_to_pandas_safe(query_non_ppm)
non_ppm_countries.country.value_counts().to_csv("non_ppm_countries.csv")
non_ppm_countries.country.unique()

# Pollutants have value of exactly 0
query_zero_pollutants = """SELECT pollutant from `bigquery-public-data.openaq.global_air_quality`
                           WHERE value = 0
                        """
zero_pollutants = open_aq.query_to_pandas_safe(query_zero_pollutants)
zero_pollutants.pollutant.value_counts().to_csv("zero_pollutant.csv")
zero_pollutants.pollutant.unique()
