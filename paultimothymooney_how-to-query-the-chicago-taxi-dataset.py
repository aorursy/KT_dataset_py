import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_taxi = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_taxi_trips")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_taxi_trips")
bq_assistant.list_tables()
bq_assistant.head("taxi_trips", num_rows=3)
bq_assistant.table_schema("taxi_trips")
query1 = """SELECT
  EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS day,
  FORMAT('%3.2f', MAX(fare)) AS maximum_fare,
  FORMAT('%3.2f', MIN(fare)) AS minimum_fare,
  FORMAT('%3.2f', AVG(fare)) AS avg_fare,
  FORMAT('%3.2f', STDDEV(fare)) AS std_dev_fare,
  COUNT(1) AS rides
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE
  trip_seconds >= 600
GROUP BY
  day
ORDER BY
  day
        """
response1 = chicago_taxi.query_to_pandas_safe(query1, max_gb_scanned=10)
response1.head(10)
query2 = """SELECT
  dropoff_community_area,
  FORMAT('%3.2f', AVG(tips)) AS average_tip,
  FORMAT('%3.2f', MAX(tips)) AS max_tip
FROM
  `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE
  dropoff_community_area IS NOT NULL
GROUP BY
  dropoff_community_area
ORDER BY
  average_tip DESC
LIMIT
  10
        """
response2 = chicago_taxi.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(10)
query3 = """SELECT
  FORMAT('%02.0fm to %02.0fm', min_minutes, max_minutes) AS minutes_range,
  SUM(trips) AS total_trips,
  FORMAT('%3.2f', SUM(total_fare) / SUM(trips)) AS average_fare
FROM (
  SELECT
    MIN(duration_in_minutes) OVER (quantiles) AS min_minutes,
    MAX(duration_in_minutes) OVER (quantiles) AS max_minutes,
    SUM(trips) AS trips,
    SUM(total_fare) AS total_fare
  FROM (
    SELECT
      ROUND(trip_seconds / 60) AS duration_in_minutes,
      NTILE(10) OVER (ORDER BY trip_seconds / 60) AS quantile,
      COUNT(1) AS trips,
      SUM(fare) AS total_fare
    FROM
      `bigquery-public-data.chicago_taxi_trips.taxi_trips`
    WHERE
      ROUND(trip_seconds / 60) BETWEEN 1 AND 90
    GROUP BY
      trip_seconds,
      duration_in_minutes )
  GROUP BY
    duration_in_minutes,
    quantile
  WINDOW quantiles AS (PARTITION BY quantile)
  )
GROUP BY
  minutes_range
ORDER BY
  Minutes_range
        """
response3 = chicago_taxi.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)