# import package with helper functions
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="openaq")

# print all the tables in this dataset (there's only one)
open_aq.list_tables()
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
non_ppm_countries = open_aq.query_to_pandas_safe(query)
non_ppm_countries.country.value_counts().head()
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
zero_pollutants =  open_aq.query_to_pandas_safe(query)
zero_pollutants.pollutant.value_counts().head