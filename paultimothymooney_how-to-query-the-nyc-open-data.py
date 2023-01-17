import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
nyc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="new_york")
bq_assistant = BigQueryHelper("bigquery-public-data", "new_york")
bq_assistant.list_tables()
bq_assistant.head("tlc_yellow_trips_2016", num_rows=10)
bq_assistant.head("citibike_trips", num_rows=10)
bq_assistant.table_schema("nypd_mv_collisions")
query1 = """SELECT
  street_name,
  borough,
  COUNT(*) AS count
FROM
  `bigquery-public-data.new_york.311_service_requests`
WHERE
  descriptor="Loud Music/Party"
  AND complaint_type="Noise - Residential"
  AND street_name != ""
GROUP BY
  street_name,
  borough
ORDER BY
  count DESC
LIMIT
  1000;
        """
response1 = nyc.query_to_pandas_safe(query1)
response1.head(50)
query2 = """SELECT
  Extract(YEAR from created_date) AS year,
  REPLACE(UPPER(complaint_type), "HEATING", "HEAT/HOT WATER") as complaint, 
  COUNT(*) AS count
FROM
  `bigquery-public-data.new_york.311_service_requests`
GROUP BY complaint, year
ORDER BY COUNT DESC
LIMIT 1000;
        """
response2 = nyc.query_to_pandas_safe(query2, max_gb_scanned=10)
response2.head(15)
query3 = """SELECT
  Latitude,
  Longitude
FROM
  `bigquery-public-data.new_york.tree_census_2015`
WHERE
  spc_common="Virginia pine";
        """
response3 = nyc.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(15)
query3 = """SELECT
  UPPER(spc_common) AS common_name,
  ROUND(latitude, 1) AS lat,
  ROUND(longitude, 1) AS long,
  COUNT(*) AS tree_count
FROM
  `bigquery-public-data.new_york.tree_census_2015`
GROUP BY
  spc_common,
  spc_latin,
  lat,
  long
ORDER BY
  tree_count DESC;
        """
response3 = nyc.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(15)
query4 = """SELECT
  UPPER(spc_common) as common_name,
  UPPER(spc_latin) as latin_name,
  COUNT(*) AS tree_count
FROM
  `bigquery-public-data.new_york.tree_census_2015`
GROUP BY
  spc_common,
  spc_latin
ORDER BY
  tree_count DESC
LIMIT
  10;
        """
response4 = nyc.query_to_pandas_safe(query4, max_gb_scanned=10)
response4.head(15)
query5 = """SELECT
  contributing_factor_vehicle_1,
  number_of_cyclist_injured,
  latitude,
  longitude
FROM
  `bigquery-public-data.new_york.nypd_mv_collisions`
WHERE
  STRPOS(contributing_factor_vehicle_1,"Animal") > 0
  AND number_of_cyclist_injured > 0;
        """
response5 = nyc.query_to_pandas_safe(query5, max_gb_scanned=10)
response5.head(15)
query6 = """SELECT
  EXTRACT(YEAR FROM timestamp) AS Year,
  COUNT(*) AS Accidents,
  SUM( number_of_persons_injured ) AS Injuries,
  SUM( number_of_persons_killed) AS Deaths
FROM
  `bigquery-public-data.new_york.nypd_mv_collisions`
GROUP BY
  Year
ORDER BY
  Year ASC;
        """
response6 = nyc.query_to_pandas_safe(query6, max_gb_scanned=10)
response6.head(15)
query7 = """SELECT
  contributing_factor_vehicle_1,
  COUNT(*) AS accidents,
  SUM(number_of_persons_injured) AS injuries,
  SUM(number_of_persons_killed) AS deaths,
  ROUND(100*(SUM( number_of_persons_killed) / COUNT(*)), 3) as pct_causing_death
FROM
  `bigquery-public-data.new_york.nypd_mv_collisions`
GROUP BY
  contributing_factor_vehicle_1
ORDER BY
  accidents DESC,
  pct_causing_death DESC;
        """
response7 = nyc.query_to_pandas_safe(query7, max_gb_scanned=10)
response7.head(15)
query8 = """SELECT
  borough,
  COUNT(*) AS accidents,
  SUM(number_of_persons_killed) AS deaths,
  ROUND(100*SUM(number_of_persons_killed)/COUNT(*), 3) as pct_deaths
FROM
  `bigquery-public-data.new_york.nypd_mv_collisions`
WHERE
  borough != ""
GROUP BY
  borough
ORDER BY
  accidents DESC;
        """
response8 = nyc.query_to_pandas_safe(query8, max_gb_scanned=10)
response8.head(15)
query9 = """SELECT
  SUM(number_of_motorist_killed) AS motorist_deaths,
  SUM(number_of_cyclist_killed) AS cyclist_deaths,
  SUM(number_of_motorist_killed) / SUM(number_of_cyclist_killed) AS ratio
FROM
  `bigquery-public-data.new_york.nypd_mv_collisions`;
        """
response9 = nyc.query_to_pandas_safe(query9, max_gb_scanned=10)
response9.head(15)
query10 = """SELECT
  start_station_name,
  end_station_name,
  COUNT(*) trips,
  ROUND(AVG(ACOS( SIN(start_station_latitude*ACOS(-1)/180)*SIN(end_station_latitude*ACOS(-1)/180) + COS(start_station_latitude*ACOS(-1)/180)*COS(end_station_latitude*ACOS(-1)/180) * COS(end_station_longitude*ACOS(-1)/180-start_station_longitude*ACOS(-1)/180) ) * 6371000)) AS distance,
  MIN(tripduration) AS fastest_trip,
  MAX(tripduration) AS slowest_trip,
  ROUND(AVG(tripduration), 3) AS average_trip
FROM
  `bigquery-public-data.new_york.citibike_trips`
WHERE
  start_station_name != end_station_name
  AND start_station_latitude != 0
  AND end_station_latitude != 0
GROUP BY
  start_station_name,
  end_station_name
HAVING
  trips > 100
ORDER BY
  distance DESC
LIMIT
  10;
        """
response10 = nyc.query_to_pandas_safe(query10, max_gb_scanned=10)
response10.head(15)
query11 = """SELECT
  start_station_name, end_station_name, avg_bike_duration, avg_taxi_duration, avg_taxi_fare
FROM (
  SELECT
    start_station_name,
    end_station_name,
    ROUND(start_station_latitude, 3) AS ss_lat,
    ROUND(start_station_longitude, 3) AS ss_long,
    ROUND(end_station_latitude, 3) AS es_lat,
    ROUND(end_station_longitude, 3) AS es_long,
    AVG(tripduration) AS avg_bike_duration,
    COUNT(*) AS bike_trips
  FROM
    `bigquery-public-data.new_york.citibike_trips`
  WHERE 
    start_station_name != end_station_name
  GROUP BY
    start_station_name,
    end_station_name,
    ss_lat,
    ss_long,
    es_lat,
    es_long
  ORDER BY
    bike_trips DESC
  LIMIT
    100 )a
JOIN (
  SELECT
    ROUND(pickup_latitude, 3) AS pu_lat,
    ROUND(pickup_longitude, 3) AS pu_long,
    ROUND(dropoff_latitude, 3) AS do_lat,
    ROUND(dropoff_longitude, 3) AS do_long,
    AVG(UNIX_SECONDS(dropoff_datetime)-UNIX_SECONDS(pickup_datetime)) AS avg_taxi_duration,
    AVG(fare_amount) as avg_taxi_fare,
    COUNT(*) AS taxi_trips
  FROM
    `bigquery-public-data.new_york.tlc_yellow_trips_2016`
  GROUP BY
    pu_lat,
    pu_long,
    do_lat,
    do_long)b
ON
  a.ss_lat=b.pu_lat AND
  a.es_lat=b.do_lat AND
  a.ss_long=b.pu_long AND
  a.es_long=b.do_long
ORDER BY
  bike_trips DESC
LIMIT 20;
        """
response11 = nyc.query_to_pandas_safe(query11, max_gb_scanned=10)
response11.head(20)
query12 = """SELECT
  COUNT(*) AS requests
FROM
  `bigquery-public-data.new_york.311_service_requests`
WHERE
  LOWER(descriptor) LIKE '%ice cream truck%';
        """
response12 = nyc.query_to_pandas_safe(query12, max_gb_scanned=10)
response12.head(20)
query13 = """SELECT
  borough,
  COUNT(descriptor) AS parties
FROM
  `bigquery-public-data.new_york.311_service_requests`
WHERE
  LOWER(descriptor) LIKE '%party%'
GROUP BY
  borough
ORDER BY
  parties DESC;
        """
response13 = nyc.query_to_pandas_safe(query13, max_gb_scanned=10)
response13.head(5)
query14 = """SELECT
  extract(DAYOFWEEK
  FROM
    created_date) AS party_day,
  borough,
  COUNT(*) AS num_parties
FROM
  `bigquery-public-data.new_york.311_service_requests`
WHERE
  LOWER(descriptor) LIKE '%party%'
GROUP BY
  party_day,
  borough
ORDER BY
  num_parties DESC;
        """
response14 = nyc.query_to_pandas_safe(query14, max_gb_scanned=10)
response14.head(30)
query15 = """SELECT
  contributing_factor_vehicle_1 AS collision_factor,
  COUNT(*) num_collisions
FROM
  `bigquery-public-data.new_york.nypd_mv_collisions`
WHERE
  contributing_factor_vehicle_1 != "Unspecified"
  AND contributing_factor_vehicle_1 != ""
GROUP BY
  1
ORDER BY
  num_collisions DESC;
        """
response15 = nyc.query_to_pandas_safe(query15, max_gb_scanned=10)
response15.head(15)
query15 = """SELECT
  on_street_name,
  SUM(number_of_persons_killed) AS deaths
FROM
  `bigquery-public-data.new_york.nypd_mv_collisions`
WHERE
  on_street_name <> ''
GROUP BY
  on_street_name
ORDER BY
  deaths DESC
LIMIT
  30;
        """
response15 = nyc.query_to_pandas_safe(query15, max_gb_scanned=10)
response15.head(30)
query16 = """SELECT
  start_station_name,
  start_station_latitude,
  start_station_longitude,
  COUNT(*) AS num_trips
FROM
  `bigquery-public-data.new_york.citibike_trips`
GROUP BY
  1,
  2,
  3
ORDER BY
  num_trips DESC
LIMIT 10;
        """
response16 = nyc.query_to_pandas_safe(query16, max_gb_scanned=10)
response16.head(30)
query17 = """SELECT
  usertype,
  CONCAT(start_station_name, " to ", end_station_name) as route,
  COUNT(*) as num_trips,
  ROUND(AVG(cast(tripduration as int64) / 60),2) as duration
FROM
  `bigquery-public-data.new_york.citibike_trips`
GROUP BY
  start_station_name, end_station_name, usertype
ORDER BY
  num_trips DESC
LIMIT 10;
        """
response17 = nyc.query_to_pandas_safe(query17, max_gb_scanned=10)
response17.head(30)
query18 = """SELECT
  CONCAT(start_station_name, " to ", end_station_name) AS route,
  COUNT(*) AS num_trips
FROM
  `bigquery-public-data.new_york.citibike_trips`
WHERE
  gender = "female"
  AND CAST(starttime AS string) LIKE '2016%'
GROUP BY
  start_station_name,
  end_station_name
ORDER BY
  num_trips DESC
LIMIT
  5;
        """
response18 = nyc.query_to_pandas_safe(query18, max_gb_scanned=10)
response18.head(30)
query19 = """SELECT
  CONCAT(start_station_name, " to ", end_station_name) AS route,
  COUNT(*) AS num_trips
FROM
  `bigquery-public-data.new_york.citibike_trips`
WHERE
  gender = "male"
  AND CAST(starttime AS string) LIKE '2016%'
GROUP BY
  start_station_name,
  end_station_name
ORDER BY
  num_trips DESC
LIMIT
  5;
        """
response19 = nyc.query_to_pandas_safe(query19, max_gb_scanned=10)
response19.head(30)
query20 = """SELECT
  TIMESTAMP_TRUNC(pickup_datetime,
    MONTH) month,
  COUNT(*) trips
FROM
  `bigquery-public-data.new_york.tlc_yellow_trips_2015`
GROUP BY
  1
ORDER BY
  1;
        """
response20 = nyc.query_to_pandas_safe(query20, max_gb_scanned=10)
response20.head(30)
query21 = """SELECT
  EXTRACT(HOUR
  FROM
    pickup_datetime) hour,
  ROUND(AVG(trip_distance / TIMESTAMP_DIFF(dropoff_datetime,
        pickup_datetime,
        SECOND))*3600, 1) speed
FROM
  `bigquery-public-data.new_york.tlc_yellow_trips_2015`
WHERE
  trip_distance > 0
  AND fare_amount/trip_distance BETWEEN 2
  AND 10
  AND dropoff_datetime > pickup_datetime
GROUP BY
  1
ORDER BY
  1;
        """
response21 = nyc.query_to_pandas_safe(query21, max_gb_scanned=10)
response21.head(30)
query22 = """SELECT
  EXTRACT(DAYOFWEEK
  FROM
    pickup_datetime) DAYOFWEEK,
  ROUND(AVG(trip_distance / TIMESTAMP_DIFF(dropoff_datetime,
        pickup_datetime,
        SECOND))*3600, 1) speed
FROM
  `bigquery-public-data.new_york.tlc_yellow_trips_2015`
WHERE
  trip_distance > 0
  AND fare_amount/trip_distance BETWEEN 2
  AND 10
  AND dropoff_datetime > pickup_datetime
GROUP BY
  1
ORDER BY
  1;
        """
response22 = nyc.query_to_pandas_safe(query22, max_gb_scanned=10)
response22.head(30)