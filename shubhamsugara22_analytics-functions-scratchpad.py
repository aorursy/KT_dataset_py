# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "san_francisco" dataset

dataset_ref = client.dataset("san_francisco", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)



# Construct a reference to the "bikeshare_trips" table

table_ref = dataset_ref.table("bikeshare_trips")



# API request - fetch the table

table = client.get_table(table_ref)



# Preview the first five lines of the table

client.list_rows(table, max_results=5).to_dataframe()
# Query to count the (cumulative) number of trips per day

num_trips_query = """

                  WITH trips_by_day AS

                  (

                  SELECT DATE(start_date) AS trip_date,

                      COUNT(*) as num_trips

                  FROM `bigquery-public-data.san_francisco.bikeshare_trips`

                  WHERE EXTRACT(YEAR FROM start_date) = 2015

                  GROUP BY trip_date

                  )

                  SELECT *,

                      SUM(num_trips) 

                          OVER (

                               ORDER BY trip_date

                               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

                               ) AS cumulative_trips

                      FROM trips_by_day

                  """



# Run the query, and return a pandas DataFrame

num_trips_result = client.query(num_trips_query).result().to_dataframe()

num_trips_result.head


# Query to track beginning and ending stations on October 25, 2015, for each bike

start_end_query = """

                  SELECT bike_number,

                      TIME(start_date) AS trip_time,

                      FIRST_VALUE(start_station_id)

                          OVER (

                               PARTITION BY bike_number

                               ORDER BY start_date

                               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING

                               ) AS first_station_id,

                      LAST_VALUE(end_station_id)

                          OVER (

                               PARTITION BY bike_number

                               ORDER BY start_date

                               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING

                               ) AS last_station_id,

                      start_station_id,

                      end_station_id

                  FROM `bigquery-public-data.san_francisco.bikeshare_trips`

                  WHERE DATE(start_date) = '2015-10-25' 

                  """



# Run the query, and return a pandas DataFrame

start_end_result = client.query(start_end_query).result().to_dataframe()

start_end_result.head()