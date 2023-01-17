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
# initialization:
# import package with helper functions 
import bq_helper
import numpy as np
import pandas as pd
# library for plotting
import matplotlib.pyplot as plt

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head("accident_2016")
# query to find out the number of accidents which 
# happen on each hour of a day
query_accident_hr = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as hour_id, COUNT(consecutive_number) as count_of_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hour_id
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accident_by_hr = accidents.query_to_pandas_safe(query_accident_hr)
accident_by_hr
list(accidents.head("vehicle_2016"))

accidents.head("vehicle_2016")["hit_and_run"]
# query to find out the state with the most 
# hit and runs
query_hnr_st = """SELECT registration_state_name,
                    COUNT("hit_and_run") as hit_n_run_count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
                     
            ORDER BY hit_n_run_count DESC
        """


hit_n_run_by_state = accidents.query_to_pandas_safe(query_hnr_st)

hit_n_run_by_state
query_all_hnr_st = """SELECT registration_state_name,
                    COUNT("registration_state_name") as total_accident_count,
                    COUNTIF(hit_and_run="Yes") as hit_n_run_count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY registration_state_name
            ORDER BY hit_n_run_count DESC
        """
all_hit_n_run_by_state = accidents.query_to_pandas_safe(query_all_hnr_st)
all_hit_n_run_by_state