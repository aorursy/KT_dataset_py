# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex5 import *

from google.cloud import bigquery



# import package with helper functions 

import bq_helper



# create a helper object for this dataset

chicago_taxi_helper = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                               dataset_name="chicago_taxi_trips")
# Your code here to find the table name

table = chicago_taxi_helper.list_tables()
# write the table name as a string below

table_name = table[0]



q_1.check()
# q_1.solution()
# your code here

from google.cloud import bigquery



client = bigquery.Client()



dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")



dataset = client.get_dataset(dataset_ref)



table_ref = dataset_ref.table(table_name)



table = client.get_table(table_ref)



client.list_rows(table, max_results=5).to_dataframe()
q_2.solution()
rides_per_year_query = """

                        SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year,

                        COUNT(*) AS num_trips

                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                        GROUP BY year

                        ORDER BY year

"""

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_year_result = client.query(rides_per_year_query, job_config=safe_config).to_dataframe()



print(rides_per_year_result)

q_3.check()
# q_3.hint()

# q_3.solution()
rides_per_month_query = """

                        SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month,

                        COUNT(*) AS num_trips

                        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                        WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017

                        GROUP BY month

                        ORDER BY month

                        

"""

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

rides_per_month_result = client.query(rides_per_month_query, job_config=safe_config).to_dataframe()



print(rides_per_month_result)

q_4.check()
#q_4.hint()

#q_4.solution()
speeds_query = """

WITH RelevantRides AS

(SELECT *

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE trip_start_timestamp > '2017-01-01' AND trip_start_timestamp < '2017-07-01'

AND trip_seconds > 0 AND trip_miles > 0

)

SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

count(*) AS num_trips,

3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

FROM RelevantRides

GROUP BY hour_of_day

ORDER BY hour_of_day

"""



# Set high max_gb_scanned because this query looks at more data

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

speeds_result = client.query(speeds_query, job_config=safe_config).to_dataframe()



print(speeds_result)

q_5.check()
# q_5.hint()

#q_5.solution()
q_6.solution()
table_ref = dataset_ref.table("taxi_trips")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the "taxi_trips" table

client.list_rows(table, max_results=200).to_dataframe()