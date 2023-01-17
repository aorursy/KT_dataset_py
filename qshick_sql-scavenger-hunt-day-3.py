# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

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
# Note that the HOUR in the EXTRACT keyword
hour_query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(hour_query)
# X axis are 00:00 to 23:59 based hours
# Y axis are numbers of accidents by hour
plt.bar(accidents_by_hour.f1_, accidents_by_hour.f0_)
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")
most_hour = accidents_by_hour.iloc[0].f1_
print("The hour that the most accidents occured in 2015 was between",
      most_hour, "and", most_hour+1)
# Let's take a quick look at the data
accidents.head("vehicle_2016")
# I will only see rows that have something in hit_and_run column
# to count numbers of hit and run's by state
hit_and_run_query = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run IS NOT NULL
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
accidents_hit_and_run = accidents.query_to_pandas_safe(hit_and_run_query)
# Change the columns to make it readable
accidents_hit_and_run.columns = ['State', 'Hit and Run']
# This is the table with hit and run data by state
accidents_hit_and_run