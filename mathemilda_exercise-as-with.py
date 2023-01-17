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



# Print names of all tables in the dataset (there are four!)

for table in tables:  

    print(table.table_id)



print(type(tables))

print(len(tables))
# Write the table name as a string below

table_name = 'taxi_trips'



# Check your answer

q_1.check()
#q_1.solution()
# Your code here

table_ref = dataset_ref.table("taxi_trips")



# API request - fetch the table

table = client.get_table(table_ref)



# Print information on all the columns in the table in the "taxi_trips" dataset

client.list_rows(table, max_results=5).to_dataframe()
#q_2.solution()
# Your code goes here

rides_per_year_query =  """SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, COUNT(1) AS num_trips

                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

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

rides_per_month_query = """SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month,

                        COUNT(1) AS num_trips

                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                        WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017

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

                   SELECT trip_start_timestamp, trip_seconds, trip_miles

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_start_timestamp > '2017-01-01' AND trip_start_timestamp < '2017-07-01'

               )

               SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day, COUNT(1) AS num_trips, 

               3600 * SUM(trip_miles) / SUM(trip_seconds) as avg_mph

               FROM RelevantRides

               WHERE trip_seconds > 0 AND trip_miles > 0

               GROUP BY hour_of_day

               ORDER BY hour_of_day

               

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, job_config=safe_config) # Your code here



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe()# Your code here



# View results

print(speeds_result)



# Check your answer

q_5.check()
#q_5.solution()
speeds_result.num_trips = speeds_result.num_trips/100000

speeds_result.set_index('hour_of_day').plot()