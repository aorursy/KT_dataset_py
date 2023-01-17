#
#our tool
import bq_helper

#prepare...
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


open_aq.list_tables()

open_aq.head("global_air_quality")

#test & sanity check
u_test = """SELECT DISTINCT unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        """
#let's go!
#task 1
units = open_aq.query_to_pandas_safe(u_test)
units

#I leave unit column to check if it worked
#LOWER clause is useful in case of potential spelling errors
query_unit = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE  LOWER(unit) != 'ppm'
            """
no_ppm = open_aq.query_to_pandas_safe(query_unit)
#answer - list of countries
no_ppm
#task 2
#same case here - I select value to check for mistakes, but you can skip it
query_value = """SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE  value = 0
            """
poll_val = open_aq.query_to_pandas_safe(query_value)

#answer - pollutants with value = 0
poll_val
