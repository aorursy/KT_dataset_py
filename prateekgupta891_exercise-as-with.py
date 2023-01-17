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

#print(dataset)

tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)

#print(table_ref)

#print(client.list_tables(dataset))

#print(dataset_ref)
# Write the table name as a string below

table_name = 'taxi_trips'



# Check your answer

q_1.check()
#q_1.solution()
# Your code here

table_ref = dataset_ref.table('taxi_trips')

table = client.get_table(table_ref)

client.list_rows(table,max_results = 5).to_dataframe()
q_2.solution()
# Your code goes here

rides_per_year_query = """

select extract(year from trip_start_timestamp) as year, count(1) as num_trips

from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

group by year

order by year

"""



# Set up the query (cancel the query if it would use too much of 

# your quota)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = client.query(rides_per_year_query,job_config = safe_config) # Your code goes here



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

with rides_per_year as

(

select extract(year from trip_start_timestamp) as year,extract(month from trip_start_timestamp) as month, count(1) as num_trips

from `bigquery-public-data.chicago_taxi_trips.taxi_trips`

group by year,month

order by year,month

)

select month,num_trips

from rides_per_year

where year = 2017

""" 



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_month_query_job = client.query(rides_per_month_query,job_config = safe_config) # Your code goes here



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

                With RelevantRides as

                  ( SELECT extract(hour from trip_start_timestamp) as hour_of_day, trip_miles,trip_seconds

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   where trip_seconds > 0 and trip_miles > 0 and

                   trip_start_timestamp > '2017-01-01' and trip_start_timestamp <'2017-07-01'

                   )

                select hour_of_day, count(1) as num_trips, 3600*(sum(trip_miles))/sum(trip_seconds) as avg_mph

               FROM RelevantRides

                group by hour_of_day

                order by hour_of_day

               """





# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query,job_config = safe_config) # Your code here



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe() # Your code here



# View results

print(speeds_result.head(5))



# Check your answer

q_5.check()
#q_5.solution()