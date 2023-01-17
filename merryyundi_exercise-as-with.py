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

for table in list(client.list_tables(dataset)):

    print(table.table_id)
# Write the table name as a string below

table_name = "taxi_trips"
# Construct a reference to the taxi_trips table

table_ref = dataset_ref.table(table_name)



# API request fetch the table

table = client.get_table(table_ref)



# Preview the first 5 lines of the taxi_trips table

client.list_rows(table,max_results=5).to_dataframe()
# Your code goes here

rides_per_year_query = """

select extract(year from trip_start_timestamp) as year, count(1) as num_trips

from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

group by year

order by year

"""



# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=1e9)

rides_per_year_query_job = client.query(rides_per_year_query,job_config=safe_config) 



# API request - run the query, and return a pandas DataFrame

rides_per_year_result = rides_per_year_query_job.to_dataframe()



# View results

print(rides_per_year_result)
# Your code goes here

rides_per_month_query = """

select extract(month from trip_start_timestamp) as month, count(1) as num_trips

from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

where extract(year from trip_start_timestamp) = 2017

group by month

order by month

""" 



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=1e9)

rides_per_month_query_job = client.query(rides_per_month_query,job_config=safe_config) 



# API request - run the query, and return a pandas DataFrame

rides_per_month_result = rides_per_month_query_job.to_dataframe() 



# View results

print(rides_per_month_result)
# Your code goes here

speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT extract(hour from trip_start_timestamp) as hour_of_day,

                   trip_miles,

                   trip_seconds

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_start_timestamp > '2017-01-01' and trip_start_timestamp < '2017-07-01'

                   and trip_miles >0

                   and trip_seconds >0

                    

               )

               SELECT hour_of_day, 

               count(1) as num_trips, 

               3600 * SUM(trip_miles) / SUM(trip_seconds) as avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=1e9)

speeds_query_job = client.query(speeds_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe() # Your code here



# View results

print(speeds_result)
# Preview the first five lines of the "taxi_trips" table

client.list_rows(table, max_results=200).to_dataframe()