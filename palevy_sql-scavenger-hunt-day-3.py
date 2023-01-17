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
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query1 = """Select Count(consecutive_number), 
                    Extract(HOUR from timestamp_of_crash)
             From `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             Group by Extract(HOUR from timestamp_of_crash)
             Order by Count(consecutive_number) desc
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)

import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Hour of Day \n (Most to least dangerous)")
print(accidents_by_hour)
import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head("vehicle_2015", selected_columns="hit_and_run", num_rows=10)
#Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state that 
#were involved in hit-and-run accidents, sorted by the number of hit and runs. 
#Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query2 = """Select registration_state_name, 
                   count(hit_and_run)
             From `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             where hit_and_run = 'Yes' 
             Group by registration_state_name
             Order by count(hit_and_run) desc
        """
hitandrun_by_state = accidents.query_to_pandas_safe(query2)
print(hitandrun_by_state)