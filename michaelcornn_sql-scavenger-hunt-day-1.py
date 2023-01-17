import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

#create a variable for the connection to the database.
openaq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq" )

#List all tables for the openaq database.
print("Listing Tables for openaq")
openaq.list_tables()
print("Listing schema for table global_air_quality")
openaq.table_schema("global_air_quality")
openaq.head("global_air_quality")
#note the DISTINCT keyword removes duplicates from the returned value.
query = "SELECT DISTINCT country FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'ppm' ORDER BY country"
noppm = openaq.query_to_pandas_safe(query)
noppm
query = "SELECT DISTINCT pollutant FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0 ORDER BY pollutant"
oppm = openaq.query_to_pandas_safe(query)
oppm