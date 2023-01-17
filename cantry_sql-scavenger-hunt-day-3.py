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
query1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS HOUR
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY HOUR
            ORDER BY HOUR ASC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by hour of Day from Morning to Night")
accidents.head("vehicle_2015")["hit_and_run"]
query2 = """SELECT count(hit_and_run) AS HITANDRUN,  
                  registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run like '%Y%'
            GROUP BY registration_state_name
            ORDER BY HITANDRUN
        """
hit_and_run_2015 = accidents.query_to_pandas_safe(query2)
print(hit_and_run_2015)
plt.bar(hit_and_run_2015.registration_state_name, hit_and_run_2015.HITANDRUN)