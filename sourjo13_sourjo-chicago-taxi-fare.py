# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

%matplotlib inline

import bq_helper

from bq_helper import BigQueryHelper
# creating an instance of the database that you can send SQL queries to later and get data back

chicago_taxi = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="chicago_taxi_trips")
print(type(chicago_taxi)) # Printing the type of object
# Printing The column headers

print(chicago_taxi.head("taxi_trips", num_rows=2).columns.tolist())
print(chicago_taxi.head("taxi_trips", num_rows=10).loc[:, ('pickup_census_tract','dropoff_census_tract')])

chicago_taxi.table_schema("taxi_trips")
query_per_year = """SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, COUNT(1) num_trips

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

        GROUP BY year

        ORDER BY year"""

rides_per_year = chicago_taxi.query_to_pandas_safe(query_per_year, max_gb_scanned=30)

#print(rides_per_year)

ax = rides_per_year.plot.bar(x = 'year', y = 'num_trips', title = 'Trips per year') 

ax.set_xlabel('Year')

ax.set_ylabel("No. of Trips")
rides_per_month_query = """

        SELECT EXTRACT(MONTH FROM trip_start_timestamp) as month, COUNT(1) num_trips

        FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`       

        WHERE EXTRACT(YEAR FROM trip_start_timestamp) = 2017 

        GROUP BY month

        ORDER BY month

"""

rides_per_month_2017 = chicago_taxi.query_to_pandas_safe(rides_per_month_query, max_gb_scanned=30)

#print(rides_per_month_2017)

rides_per_month_2017["Months"] = ["Jan", "Feb", "March", "April", "May", "June", "July", "August"]

ax = rides_per_month_2017.plot.bar(x = "Months", y = 'num_trips', title = 'Trips per month in 2017') 

ax.set_xlabel('Year')

ax.set_ylabel("No. of Trips")
speeds_query = """

        WITH RelevantRides AS

        (       

                SELECT EXTRACT(HOUR FROM trip_start_timestamp) as hour_of_day, trip_miles, trip_seconds

                FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

                WHERE trip_start_timestamp BETWEEN '2017-01-01' AND '2017-07-01' 

                        AND trip_seconds > 0 AND trip_miles > 0

        )

        SELECT hour_of_day, COUNT(1) AS num_trips, 3600 * SUM(trip_miles) / SUM(trip_seconds) AS avg_mph

        FROM RelevantRides

        GROUP BY hour_of_day

        ORDER BY hour_of_day

        """

# Set high max_gb_scanned because this query looks at more data

speeds_result = chicago_taxi.query_to_pandas_safe(speeds_query, max_gb_scanned=20)

print(speeds_result)

ax = speeds_result.plot.line(x = "hour_of_day", y = ['num_trips', 'avg_mph'], title = 'Avg trips and speed in 2017')

ax.set_xlabel("Hour of day")
query_tips = "SELECT taxi_id, avg(tips) as AVG_Tips FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` GROUP BY taxi_id ORDER BY AVG_Tips DESC"

#query_tips = "SELECT AVG(tips) as AVG FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips` WHERE taxi_id = '73c788fff8c6a38a113b2a002b98894b888233f223d0342522eddec5cf3392e7405e1d5c07cc6b0689f1dfd0e62f6454ca9572a4e5df4853bc93f5620678c497'"

tips_df = chicago_taxi.query_to_pandas_safe(query_tips, max_gb_scanned=200)
#Top 20 performers in terms of Customer Satisfaction

#print(tips_df[:20])

ax = tips_df[:20].plot.bar(x = 'taxi_id', y = 'AVG_Tips', title = 'Top 20 Performers in terms of Customer Satisfaction')

ax.set_xlabel("Taxi id")

ax.set_ylabel("Avg tips received by each Taxi")