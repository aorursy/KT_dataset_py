# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head('accident_2015')
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
# 1. Which hours of the day do the most accidents occur during (2016)?
# I'm using this table - accident_2016
# Do Not use alias for Extract() function
query_1 = """SELECT COUNT(consecutive_number)
                    , EXTRACT(HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query_1)

# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
accidents_by_hour

# 2. Which state has the most hit and runs?

# Checking the vehicle_2016 table
accidents.head('vehicle_2016')
# or use this one: accidents.table_schema('vehicle_2016')

query_2 = """SELECT registration_state_name,
                    COUNT(consecutive_number) hit_and_run_count
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             WHERE hit_and_run = 'Yes'
             GROUP BY registration_state_name
             ORDER BY hit_and_run_count DESC
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(query_2)
plt.plot(hit_and_run_by_state.hit_and_run_count)
plt.title("Number of Hit-and-Runs by States in 2016")
hit_and_run_by_state