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

# Import package with helper functions
import bq_helper
# Library for plotting
import matplotlib.pyplot as plt

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="nhtsa_traffic_fatalities")

# First Query: Accidents by hour of the day. Sort by # of accidents,
per_hour = """SELECT COUNT(consecutive_number) as Number,
                     EXTRACT(HOUR FROM timestamp_of_crash) as per_hour
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY per_hour
           ORDER BY Number DESC
           """

# First Query: Make it safe

accidents_by_hour = accidents.query_to_pandas_safe(per_hour)
print(accidents_by_hour)

# Looks like "Hour 18" has the most accidents

# Make a plot to for fun

plt.plot(accidents_by_hour)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")

# Second Query: Which state has the most hit and runs?
# Keep in mind, we're switching tables

hit_and_runs = """
                    SELECT registration_state_name, COUNT(vehicle_number) as Number
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                    WHERE hit_and_run = "Yes"
                    GROUP BY registration_state_name
                    ORDER BY Number DESC 
                    """
# Second Query: Make it safe
# Keep in mind, we use "accidents." still, while the table has changed the dataset is the same
hit_run_count = accidents.query_to_pandas_safe(hit_and_runs)
print(hit_run_count)
