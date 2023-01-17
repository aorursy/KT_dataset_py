import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
# import package helper with helper functions
import bq_helper
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head("accident_2015")
# query to find out the number of accidents which
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
           ORDER BY COUNT(consecutive_number) DESC
           """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day.head()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, stored:
plt.plot(accidents_by_day.f0_)
plt.title("Number of accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
query2 = """ SELECT COUNT(consecutive_number),
                    EXTRACT(HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
            """
        
hourly_accidents = accidents.query_to_pandas_safe(query2)
hourly_accidents
hourly_accidents.to_csv("hourly_accidents.csv")
accidents.head("vehicle_2015")
accidents.head(
        "vehicle_2015", 
        selected_columns = ["hit_and_run", "registration_state_name", "consecutive_number"],
        num_rows = 10
)
query3 = """ SELECT registration_state_name, COUNT(hit_and_run)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             WHERE hit_and_run = "Yes"
             GROUP BY registration_state_name
             ORDER BY COUNT(hit_and_run) DESC
             """
hit_run_by_state = accidents.query_to_pandas_safe(query3)
hit_run_by_state
hit_run_by_state.to_csv("hit_run_by_state.csv")
