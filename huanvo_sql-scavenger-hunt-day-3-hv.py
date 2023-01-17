# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
#check the head of accident_2016
accidents.head('accident_2015',num_rows=10)
#query that returns how many accidents happen each hour in 2015
hour_query = """SELECT COUNT(consecutive_number),
                       EXTRACT(HOUR FROM timestamp_of_crash)
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                ORDER BY COUNT(consecutive_number) DESC
             """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(hour_query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of hour \n (Most to least dangerous)")
# print accidents by hour 
accidents_by_hour
#check the head of vehicle_2016
accidents.head('vehicle_2016', num_rows=10)
#check the column hit and run 
accidents.head('vehicle_2016', selected_columns='registration_state_name, hit_and_run',num_rows=10)
#query to find the number of hit-and-run accidents in each state
hit_and_run_query = """SELECT registration_state_name, hit_and_run, COUNT(consecutive_number)
                       FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                       GROUP BY registration_state_name, hit_and_run
                       HAVING hit_and_run = 'Yes' AND
                              registration_state_name != 'Unknown' AND
                              registration_state_name != 'No Registration' AND
                              registration_state_name != 'Not Applicable'
                       ORDER BY COUNT(consecutive_number) DESC
                    """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_and_run = accidents.query_to_pandas_safe(hit_and_run_query)
hit_and_run