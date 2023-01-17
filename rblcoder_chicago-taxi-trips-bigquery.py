from google.cloud import bigquery
client = bigquery.Client()
ds_ref = client.dataset('chicago_taxi_trips', project='bigquery-public-data')
ds = client.get_dataset(ds_ref)
for d in client.list_tables(ds):

    print(d.table_id)
tbl = client.get_table(ds.table('taxi_trips'))
tbl.schema
%load_ext google.cloud.bigquery
# %%bigquery total_miles

# SELECT

#     source_year AS pickup_location,

#     SUM(trip_miles) AS trip_miles_sum

# FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

# GROUP BY pickup_location

# ORDER BY trip_miles_sum DESC

# LIMIT 10
QUERY = """

    SELECT

    pickup_location AS pickup_location,

    SUM(trip_miles) AS trip_miles_sum

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY pickup_location

ORDER BY trip_miles_sum DESC

LIMIT 10

        """
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_taxi_trips")
df = bq_assistant.query_to_pandas(QUERY)
df.plot(x='pickup_location', y='trip_miles_sum', kind='bar');
QUERY = """

    SELECT

    pickup_location,

    SUM(fare) AS fare_sum,

    SUM(trip_miles) AS trip_miles_sum

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY pickup_location

ORDER BY fare_sum DESC

LIMIT 10

        """
df_fare_miles = bq_assistant.query_to_pandas(QUERY)
df_fare_miles.plot(kind='bar')
sql = """

SELECT

    pickup_location,

    SUM(trip_seconds) AS trip_seconds_sum,

    SUM(fare) AS fare_sum

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY pickup_location

ORDER BY fare_sum DESC

LIMIT 10

"""

df_trip_seconds_fare = client.query(sql).to_dataframe()

df_trip_seconds_fare.head()
df_trip_seconds_fare.plot(kind='bar', logy=True)
sql = """

SELECT

    pickup_location,

    MAX(EXTRACT(YEAR FROM trip_start_timestamp)) AS trip_start_year_max,

    MAX(EXTRACT(YEAR FROM trip_end_timestamp)) AS trip_end_year_max,

    MIN(EXTRACT(YEAR FROM trip_start_timestamp)) AS trip_start_year_min,

    MIN(EXTRACT(YEAR FROM trip_end_timestamp)) AS trip_end_year_min,

    ROUND(AVG(EXTRACT(YEAR FROM trip_start_timestamp))) AS trip_start_year_avg,

    ROUND(AVG(EXTRACT(YEAR FROM trip_end_timestamp))) AS trip_end_year_avg

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY pickup_location

ORDER BY trip_end_year_max DESC

LIMIT 10

"""

df_trip_time_year = client.query(sql).to_dataframe()

df_trip_time_year
sql = """

SELECT

    SUM(fare) AS fare_sum,

    sum(trip_seconds) AS trip_seconds_sum,

    EXTRACT(YEAR FROM trip_end_timestamp) AS trip_end_year

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY trip_end_year

ORDER BY trip_end_year DESC

LIMIT 10

"""

df_trip_time_year = client.query(sql).to_dataframe()

df_trip_time_year
sql = """

SELECT

    avg(fare) AS fare_avg,

    avg(trip_seconds) AS trip_seconds_avg,

    EXTRACT(YEAR FROM trip_end_timestamp) AS trip_end_year

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY trip_end_year

ORDER BY trip_end_year DESC

LIMIT 10

"""

df_trip_time_year = client.query(sql).to_dataframe()

df_trip_time_year
sql = """

SELECT

    avg(fare) AS fare_avg,

    avg(trip_seconds) AS trip_seconds_avg,

    EXTRACT(MONTH FROM trip_end_timestamp) AS trip_end_month

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY trip_end_month

order by fare_avg desc

"""

df = client.query(sql).to_dataframe()

df
#Refer https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions for DAYOFWEEK

sql = """

SELECT

    avg(fare) AS fare_avg,

    avg(trip_seconds) AS trip_seconds_avg,

    avg(trip_miles) AS trip_miles_avg,

    EXTRACT(DAYOFWEEK FROM trip_end_timestamp) AS trip_end_day_of_week

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY trip_end_day_of_week

order by fare_avg desc

"""

df = client.query(sql).to_dataframe()

df[['fare_avg','trip_seconds_avg','trip_miles_avg']].corr()
#Refer https://cloud.google.com/bigquery/docs/reference/standard-sql/date_functions for DAYOFWEEK

sql = """

SELECT

    avg(fare) AS fare_avg,

    avg(trip_seconds) AS trip_seconds_avg,

    avg(trip_miles) AS trip_miles_avg,

    EXTRACT(DAYOFWEEK FROM trip_end_timestamp) AS trip_end_day_of_week,

    EXTRACT(YEAR FROM trip_end_timestamp) AS trip_end_year

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY trip_end_year, trip_end_day_of_week

order by fare_avg desc

"""

df = client.query(sql).to_dataframe()
df