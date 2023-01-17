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
accidents_by_day
import pandas as pd
head = pd.DataFrame(accidents.head('accident_2015'))
head
head['hour_of_crash']
query_1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash),
                    COUNT(consecutive_number) AS n_accidents,
                    AVG(hour_of_crash) AS hour_1                   
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY n_accidents DESC"""
accidents.estimate_query_size(query_1)
accidents_hour = accidents.query_to_pandas_safe(query_1)
accidents_hour.head()
head_2 = accidents.head("vehicle_2015")
head_2
query_2 = """SELECT registration_state_name,
                    COUNT(consecutive_number) AS total_har
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             WHERE hit_and_run = 'Yes'
             GROUP BY registration_state_name
             ORDER BY total_har DESC"""
accidents.estimate_query_size(query_2)
hit_and_runs_data = accidents.query_to_pandas_safe(query_2)
hit_and_runs_data.head()
query_3 = """SELECT registration_state_name,
                    COUNTIF(hit_and_run = 'Yes') AS total_har
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             GROUP BY registration_state_name
             ORDER BY total_har DESC"""
accidents.estimate_query_size(query_3)
hit_and_runs_d2 = accidents.query_to_pandas_safe(query_3)
hit_and_runs_d2.head()