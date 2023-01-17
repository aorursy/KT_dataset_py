# helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# The following package "helper functions " will help us interact with BigQuery

import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

open_aq.list_tables()
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT DISTINCT country
            FROM `bigquery-public-data.openaq.global_air_quality`
        """

# check how big this query will be
open_aq.estimate_query_size(query)
countries = open_aq.query_to_pandas_safe(query)
countries.country
query1 = """SELECT  country , unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
        """

countries = open_aq.query_to_pandas_safe(query1)