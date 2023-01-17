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
# Your code here to

lst = list(client.list_tables(dataset))

for x in lst:

    print(x.table_id)
# Write the table name as a string below

table_name = 'taxi_trips'



# Check your answer

q_1.check()
q_1.solution()
# Construct a reference to the "taxi_trips" table

table_ref = dataset_ref.table("taxi_trips")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "taxi_trips" table

client.list_rows(table, max_results=5).to_dataframe()
q_2.solution()


# Your code goes here

rides_per_year_query = """

                       SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

                              COUNT(1) AS num_trips

                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       GROUP by year

                       ORDER by year

                       """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 10 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = client.query(rides_per_year_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame

rides_per_year_result = rides_per_year_query_job.to_dataframe()



# View results

print(rides_per_year_result)



# Check your answer

q_3.check()
rides_per_year_query = """

                       SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, 

                              COUNT(1) AS num_trips

                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       GROUP BY year

                       ORDER BY year

                       """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

rides_per_year_query_job = client.query(rides_per_year_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

rides_per_year_result = rides_per_year_query_job.to_dataframe()

print(rides_per_year_result)
q_3.hint()

q_3.solution()
# Your code goes here

rides_per_month_query = """

                       SELECT EXTRACT(month FROM trip_start_timestamp) AS month, 

                              COUNT(1) AS num_trips

                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       where EXTRACT(YEAR FROM trip_start_timestamp) = 2017

                       GROUP BY month

                       ORDER BY month

                       """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_month_query_job = client.query(rides_per_month_query, job_config=safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

rides_per_month_result = rides_per_month_query_job.to_dataframe() # Your code goes here



# View results

print(rides_per_month_result)



# Check your answer

q_4.check()
q_4.hint()

q_4.solution()
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

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

speeds_query_job = ____ # Your code here



# API request - run the query, and return a pandas DataFrame

speeds_result = ____ # Your code here



# View results

print(speeds_result)



# Check your answer

q_5.check()
q_5.solution()
q_6.solution()