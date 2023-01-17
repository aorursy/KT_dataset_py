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
accidents.head('accident_2015')
# Your ce goes here :)



query2 = """SELECT COUNT(consecutive_number) AS Accidents, EXTRACT(HOUR FROM timestamp_of_crash) as Hour_Acc
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY Hour_Acc
                ORDER BY COUNT(consecutive_number)
                """
accidents_by_hour = accidents.query_to_pandas_safe(query2)
accidents_by_hour
# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.Accidents)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
query3 = """SELECT registration_state_name,COUNT(registration_state_name) as hit_run_count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name, hit_and_run
            HAVING hit_and_run = "Yes"
            ORDER BY COUNT(registration_state_name) DESC
        """
accidents_by_state = accidents.query_to_pandas_safe(query3)
accidents_by_state
plt.plot(accidents_by_state.hit_run_count)
plt.title("Number of Accidents by State of Hour \n (Most to least dangerous)")