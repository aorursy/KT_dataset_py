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

tables_list = list(client.list_tables(dataset))

for table in tables_list:

    print(table.table_id)

# Write the table name as a string below

table_name = 'taxi_trips'



# Check your answer

q_1.check()
#q_1.solution()
# Your code here

table_ref = dataset_ref.table(table_name)

table = client.get_table(table_ref)

client.list_rows(table,max_results=5).to_dataframe()
q_2.solution()
# Your code goes here

rides_per_year_query = """

                           SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, COUNT(1) AS num_trips

                            from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                            group by EXTRACT(YEAR FROM trip_start_timestamp)

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
# Your code goes here

rides_per_month_query = """

                             SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month, COUNT(1) AS num_trips

                            from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                            where EXTRACT(YEAR FROM trip_start_timestamp) =2017

                            group by EXTRACT(MONTH FROM trip_start_timestamp)

                            """ 



# Set up the query

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

               WITH RelevantRides AS

               (

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) as hour_of_day, trip_miles, trip_seconds

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_start_timestamp > cast('2017-01-01 00:00:00' as timestamp) and trip_start_timestamp<cast('2017-07-01 00:00:00' as timestamp)

                   and trip_seconds>0 and trip_miles>0

               )

               SELECT hour_of_day , count(1) as num_trips, 3600 * SUM(trip_miles) / SUM(trip_seconds) as avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day

               ORDER BY hour_of_day DESC

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

speeds_result =speeds_query_job.to_dataframe()



# View results

print(speeds_result)



# Check your answer

q_5.check()
#q_5.solution()