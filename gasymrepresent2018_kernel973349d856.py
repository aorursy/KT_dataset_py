
import numpy as np 
import pandas as pd 



# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "country" column where the
# "unit" column is other than "ppm"
unit_query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
NonPpmCountries = open_aq.query_to_pandas_safe(unit_query)

#Output countries which measure type is other than ppm
NonPpmCountries
# query to select all the items from the "pollutant" column where the
# "value" column is exactly 0
pollut_query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollut_0 = open_aq.query_to_pandas_safe(pollut_query)

#Output pollutants which have a value of exactly 0
pollut_0
