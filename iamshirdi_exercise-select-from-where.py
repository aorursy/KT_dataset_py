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

client.list_rows(table, max_results=5).to_dataframe()
# Query to select distinct one time no repeat countries with units of "ppm"

first_query ="""

              SELECT distinct country,unit

              FROM `bigquery-public-data.openaq.global_air_quality`

              WHERE unit = "ppm"

              """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

dry_run_config = bigquery.QueryJobConfig(dry_run=True)

dry_run_query_job = client.query(first_query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))

ONE_HUNDRED_MB = 100*1000*1000 #1e8 is float accepts only int note so int(1e8)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)

first_query_job = client.query(first_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

first_results = first_query_job.to_dataframe()



# View top few rows of results

print(first_results.head())



# Check your answer

q_1.check()
#q_1.solution()

# Query to select all columns where pollution levels are exactly 0

zero_pollution_query="select * FROM `bigquery-public-data.openaq.global_air_quality`where value=0" # Your code goes here



ONE_HUNDRED_MB = 100*1000*1000 #1e8

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)



# Set up the query

query_job = client.query(zero_pollution_query, job_config=safe_config)



# API request - run the query and return a pandas DataFrame

zero_pollution_results = query_job.to_dataframe() # Your code goes here



print(zero_pollution_results.head())



# Check your answer

q_2.check()
q_1.solution()