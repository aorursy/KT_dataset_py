# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# This notebook is the practice version following th SQL Scavenger hunt day 3
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Import package with helper functions. Only work on Kaggle for now 05/02/2018
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="nhtsa_traffic_fatalities")

# view all tables in the dataset
accidents.list_tables()

# view all the columns data definition in a table
accidents.table_schema("accident_2015")

# preview the first couple of lines of the table
#accidents.head("accident_2015", selected_columns="state_name", num_rows=10)

# query to find out the number of accidents which happen on each day of the week
query = """SELECT COUNT(consecutive_number), day_of_week FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`GROUP BY day_of_week ORDER BY COUNT(consecutive_number) DESC"""

#check how big this query will be
accidents.estimate_query_size(query)

#only run this query if it is less than 100mb
accidents_by_day=accidents.query_to_pandas_safe(query)

#library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is actualy ordered:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
accidents.table_schema("accident_2015")

# write a query to find the hours of the day the most accidents occur during
query1 = """SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash) FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` GROUP BY EXTRACT(HOUR FROM timestamp_of_crash) ORDER BY COUNT(consecutive_number) DESC"""

accidents_by_hour = accidents.query_to_pandas_safe(query1)

plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")

print(accidents_by_hour)
accidents.table_schema("vehicle_2015")
# write a query to find number of hit and run accident by each state
query2 = """SELECT COUNT(consecutive_number), registration_state_name FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` WHERE hit_and_run="Yes" GROUP BY registration_state_name ORDER BY COUNT(consecutive_number) DESC"""

accidents_by_hit_n_run = accidents.query_to_pandas_safe(query2)

plt.plot(accidents_by_hit_n_run.f0_,accidents_by_hit_n_run.registration_state_name)
plt.title("Number of Accidents by Rank of Hit and Run \n (Most to least dangerous)")

print(accidents_by_hit_n_run)