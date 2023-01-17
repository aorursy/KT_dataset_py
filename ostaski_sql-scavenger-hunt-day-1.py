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
# import the Big Query helper package
import bq_helper

# create the helper object using bigquery-public-data.openaq
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

# looking at the data
open_aq.list_tables()
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")

# Q1. Which countries use a unit other than ppm to measure any type of pollution?
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# estimate query size
open_aq.estimate_query_size(query)

# make a "safe" query and put resultset into a dataframe
non_ppm_countries = open_aq.query_to_pandas_safe(query)

# taking a look
print(non_ppm_countries)
# there are 64 countries using units other than ppm

# saving this in case we need it later
non_ppm_countries.to_csv("non_ppm_countries.csv")

# Q2. Which pollutants have a value of exactly 0?
query = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

# estimate query size
open_aq.estimate_query_size(query)

# make a "safe" query and put resultset into a dataframe
pollutant_0 = open_aq.query_to_pandas_safe(query)

# taking a look
print(pollutant_0)
# there are 7 pollutants that have at least one observation where their value is 0

# saving this in case we need it later
pollutant_0.to_csv("pollutant_0.csv")

