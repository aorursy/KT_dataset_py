# set up feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex5 import *

print("Setup is completed")
# create a "Client" object

from google.cloud import bigquery

client = bigquery.Client()



# construct a reference to the "chicago_taxi_trips" dataset

dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# your code here to find the table name

[table.table_id for table in client.list_tables(dataset)]
# write the table name as a string below

table_name = 'taxi_trips'



# check your answer

q_1.check()
# for the solution, uncomment the line below.

# q_1.solution()
# construct a reference to the "taxi_trips" table

table_ref = dataset_ref.table("taxi_trips")



# fetch the table (API request)

table = client.get_table(table_ref)



# preview the first five lines of the "global_air_quality" table

client.list_rows(table, max_results=5).to_dataframe()
q_2.solution()
# your code goes here

rides_per_year_query = """

                       SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, COUNT(1) AS num_trips

                       FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                       GROUP BY year

                       ORDER BY year DESC

                       """



# set up the query (cancel the query if it would use too much of your quota, it requiers almost 1.5GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_query_job = client.query(rides_per_year_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

rides_per_year_result = rides_per_year_query_job.to_dataframe()



# view results

print(rides_per_year_result)



# check your answer

q_3.check()
# for a hint or the solution, uncomment the appropriate line below.

# q_3.hint()

# q_3.solution()
# your code goes here

rides_per_month_query = """

                        SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month,

                               COUNT(1) AS num_trips

                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                        WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017

                        GROUP BY month

                        ORDER BY month

                        """



# set up the query (cancel the query if it would use too much of your quota, it requiers almost 1.5GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_month_query_job = client.query(rides_per_month_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

rides_per_month_result = rides_per_month_query_job.to_dataframe()



# view results

print(rides_per_month_result)



# check your answer

q_4.check()
# for a hint or the solution, uncomment the appropriate line below.

# q_4.hint()

# q_4.solution()
# your code goes here

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



# set up the query (cancel the query if it would use too much of your quota, it requiers almost 4.6GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_query_job = client.query(speeds_query, job_config=safe_config)



# API request - run the query, and return a pandas DataFrame

speeds_result = speeds_query_job.to_dataframe()



# view results

print(speeds_result)



# check your answer

q_5.check()
# for the solution, uncomment the appropriate line below.

# q_5.solution()