# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex2 import *

print("Setup Complete")
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

client.list_rows(table, selected_fields = table.schema[:5]).to_dataframe()
# Query to select countries with units of "ppm"

first_query = """

                SELECT DISTINCT country

                FROM `bigquery-public-data.openaq.global_air_quality`

                WHERE unit = 'ppm' # Your code goes here

              """

# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

first_query_job = client.query(first_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

first_results = first_query_job.to_dataframe()



# View top few rows of results

print(first_results.head())



# Check your answer

q_1.check()
#q_1.solution()
"""Some users experience a BadRequest error during this step. This can be solved by creating a

new QueryJobConfig() object separately for this query.

More explanation @ 

https://stackoverflow.com/questions/50838525/bad-request-error-while-querying-data-from-bigquery-in-a-loop/50843371"""





zero_pollution_query = """

                       SELECT *

                       FROM `bigquery-public-data.openaq.global_air_quality`

                       WHERE value = 0

                       """



query_job = client.query(zero_pollution_query, job_config=bigquery.QueryJobConfig())



zero_pollution_results = query_job.to_dataframe()



print(zero_pollution_results.head())



# Check your answer

q_2.check()
q_2.solution()