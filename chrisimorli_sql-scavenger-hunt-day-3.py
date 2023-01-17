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
# Which hours of the day do the most accidents occur during?
#
#    Return a table that has information on how many accidents occurred in each
#    hour of the day in 2015, sorted by the the number of accidents which occurred each day.
#    Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash
#    column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a 
#    chance to practice with dates. :P)
#    Hint: You will probably want to use the HOUR() function for this.


# this is my query but I don't have any idea how one can sort in the end by the
# total number of accidents on that day. :-(
hour_of_accidents_query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash), EXTRACT(DATE FROM timestamp_of_crash)
                  
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DATE FROM timestamp_of_crash), EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)  DESC
            """

hour_of_accidents = accidents.query_to_pandas_safe(hour_of_accidents_query)
hour_of_accidents.head()
# Which state has the most hit and runs?
#
#    Return a table with the number of vehicles registered in each state that were
#    involved in hit-and-run accidents, sorted by the number of hit and runs. Use either
#    the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name
#    and hit_and_run columns.

hit_and_run_query = """SELECT SUM(vehicle_number), 
                  hit_and_run, state_number
                  
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY state_number, hit_and_run
            ORDER BY SUM(vehicle_number)  DESC
            """


accidents.head('vehicle_2016')
accidents.head('vehicle_2016', selected_columns="hit_and_run")
hit_and_runs = accidents.query_to_pandas_safe(hit_and_run_query)
hit_and_runs.head()