# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

#.list_tables()
# List of Accidents in 2015
accidents.head('accident_2015').head()
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number) AS total, 
                  EXTRACT(DAY FROM timestamp_of_crash) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAY FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
print(accidents_by_day)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Return a table that has information on how many accidents occurred in each hour of the day 
# in 2015, sorted by the the number of accidents which occurred each hour. 
# Use either the accident_2015 
# or accident_2016 table for this, and the timestamp_of_crash column
query_accidents_hour = """SELECT COUNT(consecutive_number) AS total, 
                  EXTRACT(HOUR FROM timestamp_of_crash) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query_accidents_hour)
print(accidents_by_hour)
# How about plotting them?
import matplotlib.pyplot as plt

# # make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Hour \n (Most to least dangerous)")
# Return a table with the number of vehicles registered in each 
# state that were involved in hit-and-run accidents, 
# sorted by the number of hit and runs. 
# especially the registration_state_name and hit_and_run columns.
query_hits_runs = """
 SELECT count(registration_state_name) as total,registration_state_name 
 FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
 WHERE hit_and_run = 'Yes'
 group by hit_and_run,registration_state_name
 order by hit_and_run DESC 

"""
hits_runs = accidents.query_to_pandas_safe(query_hits_runs)
print(hits_runs)
import matplotlib.pyplot as plt

# # make a plot to show that our data is, actually, sorted:
plt.plot(hits_runs.registration_state_name,hits_runs.total)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 20
plt.rcParams["figure.figsize"] = fig_size
plt.xticks(rotation=45)
plt.title("Hits n Runs per state")
