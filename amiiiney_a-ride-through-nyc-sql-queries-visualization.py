import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from google.cloud import bigquery

import sql_visualization_script as viz

from sql_visualization_script import qplot

import warnings

warnings.simplefilter(action='ignore')
client= bigquery.Client()

dataset_ref= client.dataset('new_york', project= 'bigquery-public-data')
import bq_helper

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "new_york")

bq_assistant.list_tables()
query_= """

SELECT

  *

FROM

  `bigquery-public-data.new_york.tlc_yellow_trips_2016`

LIMIT

  2

  """

safe_query_job= client.query(query_)

summary_df= safe_query_job.to_dataframe()

summary_df
import sql_visualization_script as viz

from sql_visualization_script import qplot
query2= """

SELECT * 

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016` 

WHERE trip_distance < 30

AND fare_amount < 150

LIMIT 180000

"""

qplot(query2, column1='fare_amount',data='new_york')
query3= """

SELECT tolls_amount 

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016` 

WHERE trip_distance < 30

AND fare_amount < 1000



LIMIT 18000

"""

qplot(query3, 'tolls_amount', data='new_york')
query4= """

SELECT tolls_amount , fare_amount

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016`

WHERE trip_distance < 30

AND fare_amount < 450

LIMIT 18000

"""

qplot(query4, 'fare_amount','tolls_amount', data='new_york')
query5= """

SELECT pickup_datetime as Rides_per_month

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016` 

WHERE trip_distance < 30

AND fare_amount < 1000

LIMIT 180000

"""

qplot(query5, 'Rides_per_month',color='lightskyblue', data='new_york')
query6= """

SELECT *

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016` 

WHERE trip_distance < 30

AND fare_amount < 1000

LIMIT 180000

"""

qplot(query6, 'trip_distance',color='gold', data='new_york')
query7= """

SELECT *

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016` 

WHERE trip_distance < 30

AND fare_amount BETWEEN 0 AND  1000

LIMIT 180000

"""

qplot(query7, 'trip_distance','fare_amount',color='black', data='new_york')
query7= """

SELECT passenger_count as passengers

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016` 

WHERE trip_distance < 30

AND fare_amount < 1000

LIMIT 180000

"""

qplot(query7, 'passengers', color='crimson', data='new_york')
query7= """

SELECT *

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016`

WHERE trip_distance < 30

AND fare_amount < 1000

LIMIT 180000

"""

qplot(query7, column2='trip_distance',column1='passenger_count',  color='deepskyblue', data='new_york')
query8= """

SELECT * 

FROM `bigquery-public-data.new_york.tlc_yellow_trips_2016` 

WHERE trip_distance BETWEEN 14 AND 30

AND fare_amount < 1000

AND tip_amount < 40

LIMIT 180000

"""

qplot(query8, column1='tip_amount',column2='trip_distance',data='new_york')
query9= """

SELECT health as NYC_Trees_health

FROM `bigquery-public-data.new_york.tree_census_2015`

LIMIT 180000

"""

qplot(query9, 'NYC_Trees_health', data='new_york')