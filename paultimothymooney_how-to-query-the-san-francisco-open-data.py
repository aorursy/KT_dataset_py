import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
sf = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="san_francisco")
bq_assistant = BigQueryHelper("bigquery-public-data", "san_francisco")
bq_assistant.list_tables()
bq_assistant.head("film_locations", num_rows=30)
bq_assistant.table_schema("film_locations")
query0 = """SELECT
  neighborhood,
  ROUND(100*COUNTIF(STRPOS(descriptor,
        "- Not_Offensive") > 0) / COUNT(*), 2) AS not_offensive_pct,
  ROUND(100*COUNTIF(STRPOS(descriptor,
        "- Offensive") > 0) / COUNT(*), 2) AS offensive_pct,
  COUNT(*) AS total_count
FROM
  `bigquery-public-data.san_francisco.311_service_requests`
WHERE
  STRPOS(category,
    "Graffiti") > 0
GROUP BY
  neighborhood
ORDER BY
  offensive_pct DESC
LIMIT
  10;
        """
response0 = sf.query_to_pandas_safe(query0)
response0.head(10)
query1 = """SELECT
  neighborhood,
  complaint_type,
  COUNT(*) AS total_count
FROM
  `bigquery-public-data.san_francisco.311_service_requests`
WHERE
  Source="Twitter"
GROUP BY
  Neighborhood,
  complaint_type
ORDER BY
  total_count DESC
LIMIT
  30;
        """
response1 = sf.query_to_pandas_safe(query1)
response1.head(30)
query2 = """SELECT
  descriptor,
  incident_address,
  COUNT(*) AS total_count
FROM
  `bigquery-public-data.san_francisco.311_service_requests`
WHERE
  category = "MUNI Feedback" AND incident_address != "Not associated with a specific address"
GROUP BY
  incident_address,
  descriptor
ORDER BY
  total_count DESC
LIMIT 10;
        """
response2 = sf.query_to_pandas_safe(query2)
response2.head(10)
query3 = """SELECT
  call_type,
  COUNT(*) AS call_type_count
FROM
  `bigquery-public-data.san_francisco.sffd_service_calls`
WHERE
  call_type != ''
GROUP BY
  call_type
ORDER BY
  call_type_count DESC
LIMIT
  10;
        """
response3 = sf.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
query4 = """SELECT
  neighborhood_district,
  COUNTIF(call_type = "Medical Incident") AS medical_incident_count,
  COUNTIF(call_type = "Structure Fire") AS structure_fire_count,
  Count(*) as total_count
FROM
  `bigquery-public-data.san_francisco.sffd_service_calls`
GROUP BY
  neighborhood_district
ORDER BY
  total_count DESC;
        """
response4 = sf.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(10)
query5 = """SELECT
  unit_type,
  ROUND(AVG(TIMESTAMP_DIFF(on_scene_timestamp, received_timestamp, MINUTE)), 2)
    as latency,
  Count(*) as total_count
FROM
  `bigquery-public-data.san_francisco.sffd_service_calls`
WHERE
  EXTRACT(DATE from received_timestamp) = EXTRACT(DATE from on_scene_timestamp)
GROUP BY
  unit_type
ORDER BY
  latency ASC;
        """
response5 = sf.query_to_pandas_safe(query5, max_gb_scanned=10)
response5.head(10)
query6 = """SELECT
  category,
  count(*) as incident_count
FROM
  `bigquery-public-data.san_francisco.sfpd_incidents`
GROUP BY
  category
ORDER BY
  incident_count DESC
LIMIT
  10;
        """
response6 = sf.query_to_pandas_safe(query6, max_gb_scanned=10)
response6.head(10)
query7 = """SELECT
  descript,
  COUNT(*) AS incident_count_2016
FROM
  `bigquery-public-data.san_francisco.sfpd_incidents`
WHERE
  category="LARCENY/THEFT"
  AND EXTRACT(YEAR FROM timestamp) = 2016
GROUP BY
  descript
ORDER BY
  incident_count_2016 DESC
LIMIT
  10;
        """
response7 = sf.query_to_pandas_safe(query7, max_gb_scanned=10)
response7.head(10)
query8 = """SELECT
  descript,
  COUNTIF(EXTRACT(YEAR FROM timestamp) = 2016) -
  COUNTIF(EXTRACT(YEAR FROM timestamp) = 2015) AS yoy_change,
  COUNTIF(EXTRACT(YEAR FROM timestamp) = 2016) AS count_2016
FROM
  `bigquery-public-data.san_francisco.sfpd_incidents`
WHERE
  category != "NON-CRIMINAL"
GROUP BY
  descript
ORDER BY
  ABS(yoy_change) DESC
LIMIT
  10;
        """
response8 = sf.query_to_pandas_safe(query8, max_gb_scanned=10)
response8.head(10)
query9 = """SELECT
  ROUND(AVG(CAST(dbh as FLOAT64)), 2) as avg_width
FROM
  `bigquery-public-data.san_francisco.street_trees`
WHERE dbh != "";
        """
response9 = sf.query_to_pandas_safe(query9, max_gb_scanned=10)
response9.head(10)
query10 = """SELECT
  EXTRACT(YEAR from plant_date) as plantdate,
  species,
  COUNT(*) as count
FROM
  `bigquery-public-data.san_francisco.street_trees`
WHERE
  plant_date IS NOT null AND
  species != "Tree(s) ::"
GROUP BY
  plantdate, species
ORDER BY
  count desc
LIMIT 10;
        """
response10 = sf.query_to_pandas_safe(query10, max_gb_scanned=10)
response10.head(10)
query11 = """SELECT
  latitude,
  longitude,
  COUNT(*) AS count
FROM
  `bigquery-public-data.san_francisco.street_trees`
WHERE latitude IS NOT null AND longitude IS NOT null
GROUP BY
  latitude, longitude
ORDER BY
  count DESC
LIMIT
  20;
        """
response11 = sf.query_to_pandas_safe(query11, max_gb_scanned=10)
response11.head(20)

