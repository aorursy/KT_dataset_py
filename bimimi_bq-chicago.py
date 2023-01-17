# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery
client = bigquery.Client()
dataset_ref = client.dataset("chicago_taxi_trips",

                            project = "bigquery-public-data")



dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
table_ref = dataset_ref.table("taxi_trips")

table = client.get_table(table_ref)



client.list_rows(table,

                max_results = 5).to_dataframe()
# Year - trips 



query = """

            SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year,

            COUNT(1) AS num_of_trips

            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

            GROUP BY year

            ORDER BY num_of_trips DESC

            """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query_job = client.query(query,

                        job_config = safe_config)



query_results = query_job.to_dataframe()



query_results
# 2019



query_1 = """

            SELECT EXTRACT(MONTH FROM trip_start_timestamp) AS month,

            COUNT(1) AS num_of_trips

            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

            WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2019 

            GROUP BY month

            ORDER BY num_of_trips DESC

            """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query_1_job = client.query(query_1,

                          job_config = safe_config)



query_1_results = query_1_job.to_dataframe()



query_1_results

# Hour - trips num - avg spd 



query_2 = """

            WITH Relevance AS (

            SELECT EXTRACT(HOUR FROM trip_start_timestamp) AS hour_of_day,

            trip_seconds,

            trip_miles

            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

            WHERE trip_start_timestamp >= '2019-01-01' 

            AND trip_start_timestamp <= '2019-12-31'

            AND trip_seconds > 0 

            AND trip_miles > 0 

            ) 

            SELECT hour_of_day, COUNT(1) AS num_of_trips,

            (3600 * SUM(trip_miles) / SUM(trip_seconds)) AS avg_mph

            FROM Relevance

            GROUP BY hour_of_day

            ORDER BY num_of_trips DESC

            """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query_2_job = client.query(query_2,

                          job_config = safe_config)



query_2_results = query_2_job.to_dataframe()



query_2_results
# 2019 - day - hr - avg speed - trip 



query_3 = """ 

            WITH Relevant_data AS (

            

            SELECT EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS day,

            EXTRACT(MONTH FROM trip_start_timestamp) AS month,

            EXTRACT(HOUR FROM trip_start_timestamp) AS hour,

            trip_seconds,

            trip_miles

            FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

            WHERE trip_start_timestamp >'2019-01-01' 

            AND trip_start_timestamp <'2019-12-31'

            AND trip_seconds > 0 

            AND trip_miles > 0 

            )

            SELECT day, month, hour, 

            COUNT(1) AS num_of_trips, 

            (3600 * SUM(trip_miles) / SUM(trip_seconds)) AS avg_mph

                FROM Relevant_data

                GROUP BY month, day, hour

                ORDER BY num_of_trips DESC

                LIMIT 50

            """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query_3_job = client.query(query_3,

                          job_config = safe_config)



query_3_results = query_3_job.to_dataframe()



query_3_results

query_3_results.to_csv("chicago.csv", 

                       index = False)
# Visualization 

    # Helper



from bq_helper import BigQueryHelper



helper = BigQueryHelper("bigquery-public-data",

                       "chicago_taxi_trips")
df = helper.query_to_pandas(query_2)
df.plot(y = "avg_mph",

       x = "hour_of_day",

       kind = "bar");
# Fare - Trip



query_4 = """ 

        SELECT pickup_location,

        SUM(trip_miles) AS miles_sum,

        SUM(fare) AS fare_sum

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

        GROUP BY pickup_location

        ORDER BY fare_sum DESC

         LIMIT 20

        """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query_4_job = client.query(query_4,

                          job_config = safe_config)



query_4_results = query_4_job.to_dataframe()



query_4_results
df_fare_dist = helper.query_to_pandas(query_4)



df_fare_dist.plot(kind = 'bar')