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
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# Selecting everything from the dataset
Query1 ="""
        SELECT *
        FROM `bigquery-public-data.openaq.global_air_quality`
        """
        
bq_assistant = BigQueryHelper('bigquery-public-data', 'openaq')
# Make the dataset into pandas dataframe
df = bq_assistant.query_to_pandas(Query1)


# Only using pandas to answer the questions
q1 = df[(df['unit'] != 'ppm')].groupby('country').sum()
q1.reset_index(inplace=True)
q1_answer = q1['country']

q2 = df[(df['value'] == 0)].groupby('pollutant').sum()
q2.reset_index(inplace=True)
q2_answer = q2['pollutant']
print(q1_answer)
print(q2_answer)