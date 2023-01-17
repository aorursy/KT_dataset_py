import pandas as pd





pd.set_option('display.max_columns',False)
from google.cloud import bigquery
client = bigquery.Client()

dataset_ref = client.dataset("new_york_taxi_trips", project="bigquery-public-data")

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



for table in tables:  

    print(table.table_id)



    

table_ref = dataset_ref.table("tlc_green_trips_2018")

table = client.get_table(table_ref)
# This gives us an idea which columns to use, and what type of problem are we trying to solve.

table.schema
query = """

        SELECT *

        FROM `bigquery-public-data.new_york_taxi_trips.tlc_green_trips_2018`

        ORDER BY RAND()

        LIMIT 10

        """



query_job = client.query(query)

data = query_job.to_dataframe()
data
query = """

        SELECT 

        pickup_datetime as pickup,

        dropoff_datetime as dropoff,

        rate_code,

        passenger_count,

        trip_distance,

        CAST(fare_amount AS FLOAT64) as fare_amount,

        CAST(extra AS FLOAT64) as extra,

        CAST(mta_tax AS FLOAT64) as mta_tax,

        CAST(tip_amount AS FLOAT64) as tip_amount,

        CAST(tolls_amount AS FLOAT64) as tolls_amount,

        CAST(total_amount AS FLOAT64) as total_amount,

        payment_type,

        pickup_location_id as pickup_id,

        dropoff_location_id as dropoff_id

        

        FROM `bigquery-public-data.new_york_taxi_trips.tlc_green_trips_2018`

        

        LIMIT 500000

        """



query_job = client.query(query)

data = query_job.to_dataframe()
data.to_csv('downloaded.csv',index=False)