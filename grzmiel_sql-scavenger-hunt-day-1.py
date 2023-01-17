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

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()


open_aq.head("global_air_quality")

# query to select which countries use a unit other than ppm to measure any type of pollution?
query_countries_with_no_ppm = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """



# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_countries_with_no_ppm_unit = open_aq.query_to_pandas_safe(query_countries_with_no_ppm)

us_countries_with_no_ppm_unit.head(n=10)

us_countries_with_no_ppm_unit.country.value_counts().head()

query_pollutants_zero = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

pollutant_with_zero_value = open_aq.query_to_pandas_safe(query_pollutants_zero)

pollutant_with_zero_value.head(n=10)