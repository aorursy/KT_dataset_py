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
#  Query to find which hours of the day the maximum accidents took place

query = """ SELECT COUNT (consecutive_number),
                   EXTRACT (HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             GROUP BY EXTRACT (HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
        """
# Estimate the query size
accidents.estimate_query_size(query)

accidents_by_hour = accidents.query_to_pandas_safe(query)
# make a plot to show that our data is sorted in a descending manner
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hours in a day \n (Most to least number of occurences)")
# Print results
# hours are represented by 0-24
print(accidents_by_hour)
# Query to find which state has the most hit and run cases 

query = """ SELECT registration_state_name AS state, COUNT (consecutive_number) AS hitnrun_cases
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY state
            ORDER BY hitnrun_cases DESC
        """

# Estimate query size 
accidents.estimate_query_size(query)
# Import to pandas if query size <1GB

hit_run = accidents.query_to_pandas_safe(query)

plt.plot(hit_run.hitnrun_cases)
plt.title("Number of hit and run cases by state \n(Ranked by frequency)")
# The state with highest number of hit and run cases is California 
# But the highest number of hit and run cases have registration(Unknown) 
# The state has not been noted for such cases.
hit_run