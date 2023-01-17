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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "country" and "unit" column where the
# "unit" column is not "ppm"
query = """SELECT distinct country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# check how big this query will be
open_aq.estimate_query_size(query)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
otherthanppm_countries = open_aq.query_to_pandas_safe(query)
print(otherthanppm_countries.count())
otherthanppm_countries.head()
# query to select all the items from the "pollutant" and "value" column where the
# "value" column is 0
query_pollutant = """SELECT distinct pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# check how big this query will be
open_aq.estimate_query_size(query_pollutant)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
value0_pollutant = open_aq.query_to_pandas_safe(query_pollutant)
print(value0_pollutant.count())
value0_pollutant
# query to select all the items from the "country" and "unit" column where the
# "unit" column is not "ppm"
query_notppm = """
            SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality` 
            WHERE country NOT IN 
            ( SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm' )
        """
# check how big this query will be
open_aq.estimate_query_size(query_notppm)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
nonppm_countries = open_aq.query_to_pandas_safe(query_notppm)
print(nonppm_countries.count())
nonppm_countries.head()