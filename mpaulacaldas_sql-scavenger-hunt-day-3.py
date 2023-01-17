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
accidents.head("accident_2015")
query_q1 = """SELECT COUNT(consecutive_number) AS number_of_crashes,
                     EXTRACT(HOUR FROM timestamp_of_crash) AS hour_crash
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
              GROUP BY hour_crash
              ORDER BY number_of_crashes DESC"""
accidents.estimate_query_size(query_q1)
crashes_by_hour = accidents.query_to_pandas_safe(query_q1)

crashes_by_hour
accidents.head(
    "vehicle_2015", 
    selected_columns = ["hit_and_run", "registration_state_name", "consecutive_number"],
    num_rows = 10
)
query_q2 = """SELECT registration_state_name,
                     COUNTIF(hit_and_run = 'Yes') AS number_of_har
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
              GROUP BY registration_state_name
              ORDER BY number_of_har DESC"""

accidents.estimate_query_size(query_q2)
har_by_state = accidents.query_to_pandas_safe(query_q2)

har_by_state.head(10)