# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# Your code goes here :)
query1 = """SELECT distinct country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE lower(unit) != 'ppm'
            ORDER BY country
        """

non_ppm_countries = open_aq.query_to_pandas_safe(query1)

non_ppm_countries
query2 = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant
        """

value_0_pollutant = open_aq.query_to_pandas_safe(query2)

value_0_pollutant