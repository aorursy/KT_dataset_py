# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# print a list of all the tables in the accidents dataset
accidents.list_tables()
# print information on all the columns in the "accident_2015" table
# in the accidents dataset
accidents.table_schema("accident_2015")
# preview the first few rows of of the "accident_2015" table
accidents.head("accident_2015")
# query to find out the number of accidents which 
# happen on each hour of the day in 2015 and sort this by the number of accidents occurred each hour
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC """

# check how big this query will be
accidents.estimate_query_size(query)
# run the query safely and store the results in a dataframe
# only run this query if it's less than 100 MB
accidents_by_hour = accidents.query_to_pandas_safe(query, max_gb_scanned=0.1)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that the data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of hour \n (Most to least dangerous)")
# Looking into the dataframe to see which hours are most dangerous.
accidents_by_hour
# print information on all the columns in the "vehicle_2015" table
# in the accidents dataset
accidents.table_schema("vehicle_2015")
# preview the first few rows of of the "vehicle_2015" table
accidents.head("vehicle_2015")
# preview the first ten entries in the registration_state_name column of the vehicle_2015 table
accidents.head("vehicle_2015", selected_columns="registration_state_name", num_rows=10)
# preview the first fifty entries in the hit_and_run column of the vehicle_2015 table
accidents.head("vehicle_2015", selected_columns="hit_and_run", num_rows=50)
# query to find out the number of vehicles registered in each state that were involved in hit-and-run 
# accidents in the vehicle_2015 table, sorted by the number of hit and runs
query = """SELECT COUNT(hit_and_run), registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC """

# check how big this query will be
accidents.estimate_query_size(query)
# run the query safely and store the results in a dataframe
# only run this query if it's less than 100 MB
hitandruns_by_state = accidents.query_to_pandas_safe(query, max_gb_scanned=0.1)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(hitandruns_by_state.f0_)
plt.title("Number of hit and runs by State")
# The total number of hit and runs in 2015.
hitandruns_by_state.f0_.sum()
# Looking into the forst five rows of the dataframe to see which States have the most hit and runs.
hitandruns_by_state.head()