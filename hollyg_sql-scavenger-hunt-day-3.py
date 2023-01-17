# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
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
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
query_hours_most_accidents = """SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                                ORDER BY COUNT(consecutive_number) DESC"""
accidents.estimate_query_size(query_hours_most_accidents)
hours_most_accidents = accidents.query_to_pandas_safe(query_hours_most_accidents)
hours_most_accidents.head()
query_state_most_hitruns = """SELECT state_number, COUNT(consecutive_number)
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                WHERE hit_and_run = "Yes"
                                GROUP BY state_number
                                ORDER BY COUNT(consecutive_number) DESC"""

accidents.estimate_query_size(query_state_most_hitruns)
state_most_hitruns = accidents.query_to_pandas_safe(query_state_most_hitruns)
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
state_code_to_name = pd.read_csv("../input/glc_states.csv")
state_code_to_name.head()

state_most_hitruns['state_name'] = state_most_hitruns['state_number'].map(state_code_to_name.set_index('State Code')['State Name'])
state_most_hitruns.head()