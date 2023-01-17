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

tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
# Write the table name as a string below

table_name = "taxi_trips"



# Check your answer

q_1.check()
#q_1.solution()
# construct a reference to the 'taxi_trips' table

table_ref = dataset_ref.table("taxi_trips")



# api request - fetch table

table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
q_2.solution()
# Your code goes here

rides_per_year_query = """

                       WITH trip_year AS

                       (

                       SELECT EXTRACT(YEAR from trip_start_timestamp) AS year

                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       )

                       SELECT year, COUNT(1) AS num_trips

                       FROM trip_year

                       GROUP BY year

                       ORDER BY year

                       """



# Set up the query (cancel the query if it would use too much of 

# your quota)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = client.query(rides_per_year_query, job_config=safe_config) # Your code goes here



# API request - run the query, and return a pandas DataFrame

rides_per_year_result = rides_per_year_query_job.to_dataframe() # Your code goes here



# View results

print(rides_per_year_result)



# Check your answer

q_3.check()
#q_3.hint()

#q_3.solution()
# Your code goes here

rides_per_month_query = """

                        WITH trip_month AS

                        (

                        SELECT EXTRACT(MONTH from trip_start_timestamp) AS month,

                               EXTRACT(YEAR from trip_start_timestamp) AS year

                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                        )

                        SELECT month, COUNT(1) AS num_trips

                        FROM trip_month

                        WHERE year=2017

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
#q_4.hint()

#q_4.solution()
# Your code goes here

speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT trip_start_timestamp, trip_miles, trip_seconds

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_start_timestamp > '2017-01-01' and 

                         trip_start_timestamp < '2017-07-01' and

                         trip_seconds > 0 and trip_miles > 0

               )

               SELECT EXTRACT(HOUR from trip_start_timestamp) AS hour_of_day,

                      COUNT(1) AS num_trips,

                      3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, job_config=safe_config) # Your code here



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe() # Your code here



# View results

print(speeds_result)



# Check your answer

q_5.check()
#q_5.solution()