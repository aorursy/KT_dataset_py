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
# print the first couple rows of the "vehicle_2015" table
accidents.head("vehicle_2015")
#which hours of the day has the most accidents occur during 
#Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
query1 = """SELECT hour_of_crash, COUNT(vehicle_number)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        GROUP BY hour_of_crash
        ORDER BY count(vehicle_number) DESC"""

#check how big this query will be
accidents.estimate_query_size(query1)
#query returns a dataframe only if query is smaller than 0.1GB
crash_per_hour_2015 = accidents.query_to_pandas_safe(query1)
crash_per_hour_2015
#Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
query2 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour_crash, COUNT(vehicle_number)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        GROUP BY hour_crash
        ORDER BY count(vehicle_number) DESC"""

#check how big this query will be
accidents.estimate_query_size(query2)
#query returns a dataframe only if query is smaller than 0.1GB
crash_per_hour_2015a = accidents.query_to_pandas_safe(query2)
crash_per_hour_2015a
# print the first couple rows of the "vehicle_2015" table- transposed 
accidents.head("vehicle_2015").transpose()
#Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
query3 = """SELECT registration_state_name,COUNT(vehicle_number)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        WHERE hit_and_run = "Yes"
        GROUP BY registration_state_name
        ORDER BY COUNT(vehicle_number)DESC"""
        
#check how big this query will be
accidents.estimate_query_size(query3)
#query returns a dataframe only if query is smaller than 0.1GB
state_hit_and_run = accidents.query_to_pandas_safe(query3)
state_hit_and_run
#Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.
query4 = """SELECT registration_state_name,COUNTIF(hit_and_run = "Yes") AS har
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        GROUP BY registration_state_name
        ORDER BY har DESC"""
        
#check how big this query will be
accidents.estimate_query_size(query4)
#query returns a dataframe only if query is smaller than 0.1GB
state_hit_and_run2 = accidents.query_to_pandas_safe(query4)
state_hit_and_run2