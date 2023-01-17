# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import our bq_helper package
import bq_helper 
# create a helper object for Hacker News dataset
traffic = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# print a list of all the tables in the Hacker News dataset
traffic.list_tables()
# check 'accident_2015' table content
traffic.table_schema('accident_2015')
# this query looks in the 'accident_2015' table in the nhtsa_traffic_fatalities
# dataset, then gets which hour_of_crash has more consecutive_number
# For that I used a select with groupb by hour_of_crash and count consecutive_number and then order counted results as desc and select first row only.
query = """SELECT hour_of_crash, count(consecutive_number) as accidents_number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour_of_crash
            ORDER BY accidents_number DESC
            LIMIT 1
        """

# check how big this query will be
traffic.estimate_query_size(query)
# run the query and get parent id with most comments
MostAccidentsHour=traffic.query_to_pandas_safe(query)
print(MostAccidentsHour.sort_values('accidents_number', ascending=[False]).reset_index()[['hour_of_crash', 'accidents_number']])
#Checking the numbers for each hour of the day using timestamp column
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as hour, count(consecutive_number) as accidents_number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY accidents_number DESC
        """

# check how big this query will be
traffic.estimate_query_size(query)
# run the query and get parent id with most comments
MostAccidentsHour=traffic.query_to_pandas_safe(query)
df_sorted=MostAccidentsHour.sort_values('hour', ascending=[True]).reset_index()
print(df_sorted)
# library for plotting
import matplotlib.pyplot as plt
# make a plot
plt.plot(df_sorted.hour, df_sorted.accidents_number)
plt.title("Number of Accidents by Hour of Day")
traffic.table_schema('vehicle_2015')
traffic.head('vehicle_2015',selected_columns='hit_and_run')
#Checking the states for hit and run
query = """SELECT  COUNTIF(v.hit_and_run = 'Yes')as hit_and_run_number, v.state_number, a.state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` as v
            JOIN `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` as a
            ON v.state_number=a.state_number
            GROUP BY v.state_number,a.state_name
            ORDER BY hit_and_run_number DESC
        """

# check how big this query will be
traffic.estimate_query_size(query)
hit_run=traffic.query_to_pandas_safe(query)
print(hit_run.sort_values('hit_and_run_number', ascending=[False]).reset_index()[['state_name', 'hit_and_run_number']])