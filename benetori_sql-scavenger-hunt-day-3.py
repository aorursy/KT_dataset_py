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
# Your code goes here :)
accidents_by_hour_query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as hour,
                             COUNT(consecutive_number) as consecutive_num_cnt
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                             GROUP BY hour
                             ORDER BY hour
                          """

accidents_by_hour = accidents.query_to_pandas_safe(accidents_by_hour_query)

print(accidents_by_hour)

import matplotlib.pyplot as plt

plt.figure(figsize=(13, 6))

plt.xlabel("hour of the day")
plt.ylabel("consecutive_num_cnt")
plt.plot(accidents_by_hour.hour,accidents_by_hour.consecutive_num_cnt,marker=".",c="r")
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
HR_registration_state_query = """select registration_state_name, count(hit_and_run) as HR_total
                                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                                 WHERE hit_and_run = 'Yes'
                                 GROUP BY registration_state_name
                                 ORDER BY HR_total DESC
                              """

HR_registration_state = accidents.query_to_pandas_safe(HR_registration_state_query)

print(HR_registration_state)
fig = plt.figure(figsize=(12, 12))
plt.barh(HR_registration_state.registration_state_name, HR_registration_state.HR_total)