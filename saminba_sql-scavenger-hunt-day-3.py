# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.head('accident_2015')


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
#plt.plot(accidents_by_day.f0_)
#plt.plot(accidents_by_day.f0_,'bo')
#plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
plt.bar(accidents_by_day.f1_,accidents_by_day.f0_)
print(accidents_by_day)
#Which hours of the day do the most accidents occur during? 
myquery1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(myquery1)
print(accidents_by_hour.head())
plt.bar(accidents_by_hour.f1_,accidents_by_hour.f0_)
#accidents.head('vehicle_2015')
tempquery="""SELECT distinct hit_and_run 
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             """
hitandrun=accidents.query_to_pandas_safe(tempquery)
hitandrun.head()

tempquery="""SELECT distinct registration_state_name
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             WHERE hit_and_run="Yes" 
             """
all_hit_and_run_states=accidents.query_to_pandas_safe(tempquery)
print(all_hit_and_run_states)
# Which state has the most hit and runs?
myquery2 = """SELECT registration_state_name,  COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes" AND registration_state_name!="Unknown"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """

state_accident=accidents.query_to_pandas_safe(myquery2)
state_accident.head()
import numpy as np
y_pos = np.arange(len(state_accident.registration_state_name))
plt.barh(y_pos[::-1],state_accident.f0_)
plt.yticks(y_pos[::-1], state_accident.registration_state_name)
# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print( "Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 15
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size