# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
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
#query = """SELECT COUNT(consecutive_number), 
#                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
#            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
#            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
#            ORDER BY COUNT(consecutive_number) DESC
#        """
# Which hours of the day do the most accidents occur during?
query1 = """ SELECT COUNT(consecutive_number),
                    EXTRACT(HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash) 
             ORDER BY COUNT(consecutive_number) DESC
             """
accidents.estimate_query_size(query1)

accidents_by_hour = accidents.query_to_pandas_safe(query1)
print(accidents_by_hour)
# library for plotting
#import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
#plt.plot(accidents_by_day.f0_)
#plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")

import matplotlib.pyplot as plt

plt.plot(accidents_by_hour.f0_)
plt.title("Number of accidents by rank of hour")
#  Which state has the most hit and runs?
#  Return a table with the number of vehicles registered in each state that were involved in hit-and-run 
#  accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for
#  this, especially the registration_state_name and hit_and_run columns.

query2 = """ SELECT COUNT(consecutive_number),
                    registration_state_name
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             WHERE hit_and_run = "Yes"
             GROUP BY registration_state_name
             ORDER BY COUNT(consecutive_number) DESC
         """ 

 # WHERE hit_and_run = "Yes"

hit_and_runs = accidents.query_to_pandas_safe(query2)

print(hit_and_runs)