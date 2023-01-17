import bq_helper
from bq_helper import BigQueryHelper
import time

chicago_taxi = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_taxi_trips")

creae_model_sql = """
CREATE MODEL
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
OPTIONS
  ( model_type='linear_reg',
    ls_init_learn_rate=.15,
    l1_reg=1,
    max_iterations=5 ) AS
SELECT
    fare label,
    trip_seconds,
    trip_miles
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
"""
res = chicago_taxi.client.query(creae_model_sql)
while not res.done():
    time.sleep(0.1)
res.result()
