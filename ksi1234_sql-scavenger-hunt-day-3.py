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


# which hour of the day has the most crashes
# Method 1
queryQ1_1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(queryQ1_1)
print(accidents_by_hour)
# which hour of the day has the most crashes
# Method 2

queryQ1_2 = """SELECT hour_of_crash, COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour_of_crash
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour_cheat = accidents.query_to_pandas_safe(queryQ1_2)
print(accidents_by_hour_cheat)
# which hour of the day has the most crashes
# 2016 data
queryQ1_3 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour_2016 = accidents.query_to_pandas_safe(queryQ1_3)
print(accidents_by_hour_2016, accidents_by_hour)
plt.plot(accidents_by_hour.f0_)
# which states have the most vehicles involved in hit and run
queryQ2 = """SELECT COUNT(hit_and_run), 
                  registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(queryQ2)
print(hit_and_run_by_state)