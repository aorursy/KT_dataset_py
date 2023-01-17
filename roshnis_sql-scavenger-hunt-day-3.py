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
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
#Listing the first few rows of the table accident_2015
accidents.head("accident_2015")
#query to find which hour do most accidents occur
query = """SELECT COUNT(consecutive_number) as accident_count,EXTRACT(HOUR FROM timestamp_of_crash) as accident_hour
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY accident_hour
        ORDER BY accident_count DESC
"""
#Executing the query and printing the result
accidents_by_hour = accidents.query_to_pandas_safe(query)
print(accidents_by_hour)
#Ploting the result in a line plot with titles, label and custom x values
import matplotlib.pyplot as plt
plt.title('Accidents by Hour in 2015')
plt.xlabel('Hour')
plt.ylabel('No of Accidents')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
plt.plot(accidents_by_hour.accident_count)
#Listing first few rows of Vehicle_2015 table
accidents.head('vehicle_2015')
#query to find which state has most hit and runs
query = """SELECT registration_state_name as state , count(hit_and_run) as hit_n_runs
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             WHERE hit_and_run = "Yes"
             GROUP BY state
             ORDER BY hit_n_runs desc"""

#Executing the query and displaying the result
hit_n_runs = accidents.query_to_pandas_safe(query)
print(hit_n_runs)
hit_n_runs.to_csv("hit_and_runs.csv")