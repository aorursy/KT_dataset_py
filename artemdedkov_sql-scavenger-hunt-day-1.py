# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
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
unit_query = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'PPM'
                """
unique_country = open_aq.query_to_pandas_safe(unit_query)
unique_country.head()

pollutant_query = """ select distinct pollutant
                    from `bigquery-public-data.openaq.global_air_quality`
                    where value = 0
                    """
unique_pollutant = open_aq.query_to_pandas_safe(pollutant_query)
unique_pollutant.head()