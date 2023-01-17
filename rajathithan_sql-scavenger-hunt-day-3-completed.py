# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.table_schema(table_name="accident_2015")
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
# Your code goes# query to find out the number of accidents which 
# happen on each hour of the day
hour_query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as Hour_of_Accident,
            COUNT(consecutive_number) as Accidents_Count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour_of_Accident
            ORDER BY Accidents_Count DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(hour_query)
print(accidents_by_hour)
# 6 PM has the highest number of accidents. 
# 4 AM has the lowest number of accidents.
# library for plotting
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
# make a plot to show that our data is, actually, sorted:
plt.bar(accidents_by_hour.Hour_of_Accident,accidents_by_hour.Accidents_Count,align='center', alpha=0.5)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
# Your code goes# query to find out the number of accidents which 
# happen on each hour of the day
state_query = """SELECT registration_state_name as State,
            COUNT(hit_and_run) as hitRunCount
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY State
            ORDER BY hitRunCount DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hitandrun_by_state = accidents.query_to_pandas_safe(state_query)
print(hitandrun_by_state)
# It is hard to determine as there is an unknown category.
# California has the highest hit and run accidents.
# District of colombia has the lowest hit and run accidents