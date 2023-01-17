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
# List all the tables in the dataset

tables = list(client.list_tables(dataset))



# Print names of all tables in the dataset (there is only one!)

for table in tables:  

    print(table.table_id)
# Write the table name as a string below

table_name = "taxi_trips"



# Check your answer

q_1.check()
#q_1.solution()
# Your code here

table_ref = dataset_ref.table("taxi_trips")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "transactions" table

client.list_rows(table, max_results=5).to_dataframe()
q_2.solution()
# Your code goes here

rides_per_year_query = """

                       SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS Year, COUNT(1) AS num_trips

                       from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       GROUP BY year

                       ORDER BY year

                       """



# Set up the query (cancel the query if it would use too much of 

# your quota)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = client.query(rides_per_year_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

rides_per_year_result = rides_per_year_query_job.to_dataframe()



# View results

print(rides_per_year_result)



# Check your answer

q_3.check()
#q_3.hint()

#q_3.solution()
rides_per_month_query = """

                       SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month, COUNT(1) AS num_trips

                       from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017

                       GROUP BY month

                       ORDER BY month

                       """



# Set up the query (cancel the query if it would use too much of 

# your quota)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_month_query_job = client.query(rides_per_month_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

rides_per_month_result = rides_per_month_query_job.to_dataframe()



# View results

print(rides_per_month_result)



# Check your answer

q_4.check()
#q_4.hint()

#q_4.solution()
# Your code goes here

speeds_query = """

               WITH Trips as(

               SELECT EXTRACT(HOUR from trip_start_timestamp) AS hour_of_day, trip_miles, trip_seconds

               FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

               where trip_seconds > 0 and trip_miles > 0 and trip_start_timestamp > '2017-01-01' and trip_start_timestamp < '2017-07-01'

               )

               SELECT hour_of_day, COUNT(1) as num_trips, 3600 * sum(trip_miles) / sum(trip_seconds) as avg_mph

               from Trips

               group by hour_of_day

               order by hour_of_day

               """



# Set up the query (cancel the query if it would use too much of 

# your quota)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe()



# View results

print(speeds_result)



# Check your answer

q_5.check()
#q_5.solution()
