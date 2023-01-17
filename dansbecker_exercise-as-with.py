# Set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex5 import *

print("Setup Complete")
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "chicago_taxi_trips" dataset

dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Your code here to find the table name
# Write the table name as a string below

table_name = ____



# Check your answer

q_1.check()
#q_1.solution()
# Your code here
# Check your answer (Run this code cell to receive credit!)

q_2.solution()
# Your code goes here

rides_per_year_query = """____"""



# Set up the query (cancel the query if it would use too much of 

# your quota)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = ____ # Your code goes here



# API request - run the query, and return a pandas DataFrame

rides_per_year_result = ____ # Your code goes here



# View results

print(rides_per_year_result)



# Check your answer

q_3.check()
#q_3.hint()

#q_3.solution()
# Your code goes here

rides_per_month_query = """____""" 



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_month_query_job = ____ # Your code goes here



# API request - run the query, and return a pandas DataFrame

rides_per_month_result = ____ # Your code goes here



# View results

print(rides_per_month_result)



# Check your answer

q_4.check()
#q_4.hint()

#q_4.solution()
# Your code goes here

speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT ____

                   FROM ____

                   WHERE ____

               )

               SELECT ______

               FROM RelevantRides

               GROUP BY ____

               ORDER BY ____

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = ____ # Your code here



# API request - run the query, and return a pandas DataFrame

speeds_result = ____ # Your code here



# View results

print(speeds_result)



# Check your answer

q_5.check()
#q_5.solution()