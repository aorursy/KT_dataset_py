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



print (dataset)
# Your code here to find the table name

tables = list(client.list_tables(dataset))
# Write the table name as a string below

#table_name = ''

for table in tables:

    table_name = table.table_id

    

print (table_name)



# Check your answer

q_1.check()
#q_1.solution()
# Construct a reference to the "taxi_trips" table

table_ref = dataset_ref.table("taxi_trips")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "taxi_trips" table

client.list_rows(table, max_results = 5).to_dataframe()
#q_2.solution()
# Your code goes here

rides_per_year_query = """

                       WITH trips AS

                       (

                           SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year

                           FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       )

                       SELECT year, COUNT(1) AS num_trips

                       FROM trips

                       GROUP BY year

                       ORDER BY year

                       """



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = client.query(rides_per_year_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

rides_per_year_result =  rides_per_year_query_job.to_dataframe()



# View results

print(rides_per_year_result)



# Check your answer

q_3.check()
#q_3.hint()

#q_3.solution()
# Your code goes here

rides_per_month_query = """

                        WITH rides AS

                        (

                            SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month

                            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                            WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017

                        )

                        SELECT month, COUNT(1) AS num_trips

                        FROM rides

                        GROUP BY month

                        ORDER BY month

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

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,  

                          trip_seconds, 

                          trip_miles

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_seconds > 0 and 

                         trip_miles > 0 and 

                         trip_start_timestamp > '2017-01-01' and

                         trip_start_timestamp < '2017-07-01'

               )

               SELECT hour_of_day, 

                      COUNT(1) AS num_trips, 

                      3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe()



# View results

print(speeds_result)



# Check your answer

q_5.check()
#q_5.solution()
# Construct a reference to the "taxi_trips" table

table_ref = dataset_ref.table("taxi_trips")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "taxi_trips" table

client.list_rows(table, max_results=200).to_dataframe()
q_6.solution()