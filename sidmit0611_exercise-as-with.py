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

for i in tables:
    print(i.table_id)
table_ref = dataset_ref.table('taxi_trips')

table = client.get_table(table_ref)
# Write the table name as a string below
table_name = 'taxi_trips'

# Check your answer
q_1.check()
# Your code here

client.list_rows(table, max_results = 5).to_dataframe()
# Your code goes here
rides_per_year_query = """  
                        SELECT EXTRACT(YEAR FROM trip_start_timestamp) as year, COUNT(unique_key) as num_trips
                        from `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                        GROUP BY year
                        ORDER BY year
                       """

# Set up the query (cancel the query if it would use too much of 
# your quota)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
rides_per_year_query_job = client.query(rides_per_year_query, job_config = safe_config) # Your code goes here

# API request - run the query, and return a pandas DataFrame
rides_per_year_result = rides_per_year_query_job.to_dataframe() # Your code goes here

# View results
print(rides_per_year_result)

# Check your answer
q_3.check()
# # Your code goes here
# rides_per_month_query = """

#                         WITH table1 AS
#                         (
#                         SELECT EXTRACT(YEAR FROM trip_start_timestamp) as year, EXTRACT(MONTH FROM trip_start_timestamp) as month, 
#                         unique_key from
#                         FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
#                         )
                        
#                         SELECT month, COUNT(unique_key) as num_trips
#                         from table1
#                         WHERE year = 2017
#                         GROUP BY month
#                         ORDER BY month
                        
#                         """ 

rides_per_month_query = """
                        SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month, 
                               COUNT(*) AS num_trips
                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                        WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017
                        GROUP BY month
                        ORDER BY month
                     """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
rides_per_month_query_job = client.query(rides_per_month_query, job_config = safe_config) # Your code goes here

# API request - run the query, and return a pandas DataFrame
rides_per_month_result =  rides_per_month_query_job.to_dataframe() # Your code goes here


# View results
print(rides_per_month_result)

# Check your answer
q_4.check()
# Your code goes here
speeds_query = """
               WITH RelevantRides AS
               (
                   SELECT unique_key, trip_start_timestamp, trip_seconds, trip_miles
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE trip_seconds > 0 and trip_miles > 0 and 
                   trip_start_timestamp > '2017-01-01' 
                   and trip_start_timestamp < '2017-07-01'
               )
               SELECT EXTRACT(HOUR from trip_start_timestamp) as hour_of_day, count(*) as num_trips,
               3600 * SUM(trip_miles) / SUM(trip_seconds) as avg_mph
               FROM RelevantRides
               GROUP BY hour_of_day
               ORDER BY hour_of_day
               """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
speeds_query_job = client.query(speeds_query, job_config = safe_config) # Your code here

# API request - run the query, and return a pandas DataFrame
speeds_result = speeds_query_job.to_dataframe() # Your code here

# View results
print(speeds_result)

# Check your answer
q_5.check()