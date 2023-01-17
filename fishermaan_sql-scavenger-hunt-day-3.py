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
# HOUR(): Returns the hour of a TIMESTAMP as an integer between 0 and 23.
query = """SELECT EXTRACT( HOUR FROM timestamp_of_crash) AS hours,
            COUNT(consecutive_number) as nums 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hours
            ORDER BY COUNT(consecutive_number) DESC
        """

accidents_by_hours = accidents.query_to_pandas_safe(query)

plt.scatter(accidents_by_hours.hours, accidents_by_hours.nums)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")

accidents_by_hours.head()
accidents.table_schema("vehicle_2015")

# Your code goes here :)
# HOUR(): Returns the hour of a TIMESTAMP as an integer between 0 and 23.
query = """SELECT registration_state_name AS states,
            COUNT(hit_and_run) as nums 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY states
            ORDER BY nums DESC
        """


hit_and_run = accidents.query_to_pandas_safe(query)

plt.bar(hit_and_run.states[0:5],hit_and_run.nums[0:5])

plt.title("Top 5 Hit and runs \n in different states")

hit_and_run