# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# check what distinct units are used in the dataset

query = """SELECT DISTINCT unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            """
open_aq.query_to_pandas_safe(query)
# list countries not using ppm as a unit

query2 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
            """
country_no_ppm = open_aq.query_to_pandas_safe(query2)
country_no_ppm
query3 = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            """
all_countries = open_aq.query_to_pandas_safe(query3)

country_no_ppm.equals(all_countries)

# list pollutants with a 0 value

query4 = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0;
            """
open_aq.query_to_pandas_safe(query4)