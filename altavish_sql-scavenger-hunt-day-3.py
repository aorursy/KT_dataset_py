# import package with helper functions 
import bq_helper as bq

# create a helper object for this dataset
accidents = bq.BigQueryHelper(active_project="bigquery-public-data",dataset_name="nhtsa_traffic_fatalities")

# print the first couple rows of the "accidents_2015" table
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
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
#Print the first couple rows of the "accidents_2015" table
accidents.head("accident_2015", selected_columns = "hour_of_crash")
#Task1:Which hours of the day do the most accidents occur during?
acc = """SELECT MAX(consecutive_number), hour_of_crash 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
GROUP BY hour_of_crash ORDER BY MAX(consecutive_number) DESC"""

accidents.query_to_pandas_safe(acc).head()
#Insight: the 21st hour (9:00 PM) has the most number of accidents in 2015. 
#Followed by 11:00 pm, 11:00 am, 10:00 am, and 6:00 pm.
#Task2:how many accidents occurred in each hour of the day in 2015?
acctwo = """SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash) AS hour
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY hour ORDER BY hour"""

accidents.query_to_pandas_safe(acctwo)
#Insights: The number of accidents that occurred in 2015 have been ordered per  0-23 hours.
#Task3:Which state has the most hit and runs?
state = """SELECT registration_state_name as state, COUNT(hit_and_run) AS counts 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
GROUP BY state ORDER by counts DESC """


accidents.query_to_pandas_safe(state).head()
#Insights: Texas has the most hit and run cases
#Task4: number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. 
numveh = """SELECT  COUNT(consecutive_number) 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
WHERE hit_and_run = 'Yes'
ORDER BY COUNT(hit_And_run)"""


accidents.query_to_pandas_safe(numveh)
#Insight: 1809 cases were involved in hit and run.