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

print(f'Number of tables: {len(tables)}\n')



print('Tables: ')

for table in tables:

    print(table.table_id)
# Write the table name as a string below

table_name = 'taxi_trips'



# Check your answer

q_1.check()
#q_1.solution()
table_ref = dataset_ref.table(table_name)

table = client.get_table(table_ref)



client.list_rows(table, max_results = 5).to_dataframe()
q_2.solution()
# Your code goes here

rides_per_year_query = """SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year,

                                 COUNT(1) AS num_trips

                          FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                          GROUP BY year

                          ORDER BY year"""



# Set up the query (cancel the query if it would use too much of 

# your quota)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = client.query(rides_per_year_query, safe_config) # Your code goes here



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

                          ORDER BY month"""



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_month_query_job = client.query(rides_per_month_query, safe_config) # Your code goes here



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

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

                          COUNT(1) as num_trips,

                          3600*SUM(trip_miles)/SUM(trip_seconds) AS avg_mph

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_seconds > 0 AND

                         trip_miles > 0 AND

                         trip_start_timestamp > '2017-01-01' AND

                         trip_start_timestamp < '2017-07-01'

                   GROUP BY hour_of_day

               )

               SELECT hour_of_day, num_trips, avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day, num_trips, avg_mph

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, safe_config) # Your code here



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe() # Your code here



# View results

print(speeds_result)



# Check your answer

q_5.check()
# q_5.solution()
q_6.solution()
# Your code goes here

speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

                          COUNT(1) as num_trips,

                          3600*SUM(trip_miles)/SUM(trip_seconds) AS avg_mph

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_seconds > 0 AND

                         trip_miles > 0 AND

                         trip_start_timestamp = '2017-01-01'

                   GROUP BY hour_of_day

               )

               SELECT hour_of_day, num_trips, avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day, num_trips, avg_mph

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, safe_config)

speeds_result = speeds_query_job.to_dataframe()

print(speeds_result)
# Your code goes here

speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

                          COUNT(1) as num_trips,

                          3600*SUM(trip_miles)/SUM(trip_seconds) AS avg_mph

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_seconds > 0 AND

                         trip_miles > 0 AND

                         trip_start_timestamp = '2017-07-01'

                   GROUP BY hour_of_day

               )

               SELECT hour_of_day, num_trips, avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day, num_trips, avg_mph

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, safe_config)

speeds_result = speeds_query_job.to_dataframe()

print(speeds_result)
speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day, 

                          trip_miles, 

                          trip_seconds

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_start_timestamp > '2017-01-01' AND 

                         trip_start_timestamp < '2017-07-01' AND 

                         trip_seconds > 0 AND 

                         trip_miles > 0

               )

               SELECT hour_of_day, 

                      COUNT(1) AS num_trips, 

                      3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day

               ORDER BY hour_of_day

               """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, job_config=safe_config)

speeds_result = speeds_query_job.to_dataframe()

print(speeds_result)
# WHERE trip_start_timestamp BETWEEN '2017-01-01' AND '2017-07-01'

# Your code goes here

speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

                          COUNT(1) as num_trips,

                          3600*SUM(trip_miles)/SUM(trip_seconds) AS avg_mph

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_seconds > 0 AND

                         trip_miles > 0 AND

                         trip_start_timestamp BETWEEN '2017-01-01' AND '2017-07-01'

                   GROUP BY hour_of_day

               )

               SELECT hour_of_day, num_trips, avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day, num_trips, avg_mph

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, safe_config)

speeds_result = speeds_query_job.to_dataframe()

print(speeds_result)
speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

                          COUNT(1) as num_trips,

                          3600*SUM(trip_miles)/SUM(trip_seconds) AS avg_mph

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_seconds > 0 AND

                         trip_miles > 0 AND

                         trip_start_timestamp BETWEEN '2017-01-01' AND '2017-07-01'

                   GROUP BY hour_of_day

               )

               SELECT hour_of_day, num_trips, avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day, num_trips, avg_mph

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, safe_config)

speeds_result = speeds_query_job.to_dataframe()

print(speeds_result)
speeds_query = """

               WITH RelevantRides AS

               (

                   SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

                          COUNT(1) as num_trips,

                          3600*SUM(trip_miles)/SUM(trip_seconds) AS avg_mph

                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                   WHERE trip_seconds > 0 AND

                         trip_miles > 0 AND

                         trip_start_timestamp BETWEEN '2017-01-01 00:00:01' AND '2017-06-30 23:59:59'

                   GROUP BY hour_of_day

               )

               SELECT hour_of_day, num_trips, avg_mph

               FROM RelevantRides

               GROUP BY hour_of_day, num_trips, avg_mph

               ORDER BY hour_of_day

               """



# Set up the query

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, safe_config)

speeds_result = speeds_query_job.to_dataframe()

print(speeds_result)

q_5.check()