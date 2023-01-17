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
# Which hour of the day do the most accidents occur during?
# Return a table with count for each hour in 2016.

query1 = """ SELECT COUNT(consecutive_number)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
         """

accidents_by_hour = accidents.query_to_pandas_safe(query1)
accidents_by_hour
# Which state had the most hit and runs in 2016?

query2 = """ SELECT registration_state_name, COUNT(hit_and_run) AS number
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             WHERE hit_and_run = "Yes"
             GROUP BY registration_state_name
             ORDER BY number DESC
         """

state_hit_and_run = accidents.query_to_pandas_safe(query2)
state_hit_and_run