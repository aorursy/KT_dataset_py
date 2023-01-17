# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex2 import *

print("Setup Complete")





# import package with helper functions 

import bq_helper



# create a helper object for this dataset

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="openaq")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "global_air_quality" table

table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
# Your Code Goes Here

first_query = """SELECT DISTINCT country

                FROM `bigquery-public-data.openaq.global_air_quality`

                WHERE unit = 'ppm'

            """



first_results = open_aq.query_to_pandas_safe(first_query)



# View top few rows of results

print(first_results.head())



q_1.check()
# Your Code Goes Here



zero_pollution_query = """SELECT *

                        FROM `bigquery-public-data.openaq.global_air_quality`

                        WHERE value = 0

                        """



zero_pollution_results = open_aq.query_to_pandas_safe(zero_pollution_query)



print(zero_pollution_results.head())



q_2.check()