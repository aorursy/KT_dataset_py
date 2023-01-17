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
query1 = """SELECT COUNT(consecutive_number) AS Accidents,
            EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC"""

query_1 = accidents.query_to_pandas_safe(query1)
print(query_1)

import matplotlib.pyplot as plt
plt.plot(query_1.Accidents,query_1.f0_)
plt.xlabel("Number of Accidents")
plt.ylabel("HOURS")
plt.title("Hourly Accident Analysis")

query2 = """SELECT registration_state_name AS Registered_State, COUNT(hit_and_run) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY Registered_State 
            ORDER BY COUNT(hit_and_run)"""

query_2 = accidents.query_to_pandas_safe(query2)
print(query_2)

plt.plot(query_2.Registered_State, query_2.f0_)
plt.title("Hit and run Cases")