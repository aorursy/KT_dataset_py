# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex2 import *



from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "openaq" dataset

dataset_ref = client.dataset("openaq", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# List all the tables in the "openaq" dataset

tables = list(client.list_tables(dataset))



table_ref = dataset_ref.table("global_air_quality")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()




# Your Code Goes Here

first_query ="""SELECT country

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE unit = 'ppm'

            """

query_job = client.query(first_query)

first_results = query_job.to_dataframe()



# View top few rows of results

print(first_results.head())



q_1.check()
#q_1.solution()
# Your Code Goes Here



zero_pollution_query = """SELECT *

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE value=0

            """



query_job = client.query(zero_pollution_query)



zero_pollution_results = query_job.to_dataframe()



print(zero_pollution_results.head())



q_2.check()
q_2.solution()