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
"""
Only looks at head, no need for anything else I don't figure.
DATA!
"""


import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper

Query1 ="""
        SELECT country, location, unit
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE unit != 'ppm'
        """
        
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
df = bq_assistant.query_to_pandas(Query1).groupby('country').sum()
df.reset_index(inplace=True)
df_answer = df['country']
Query2 ="""
        SELECT DISTINCT pollutant
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE value = 0
        """
        
bq_assistant2 = BigQueryHelper('bigquery-public-data', 'openaq')
df2 = bq_assistant2.query_to_pandas(Query2)

print(df_answer)
print(df2)
