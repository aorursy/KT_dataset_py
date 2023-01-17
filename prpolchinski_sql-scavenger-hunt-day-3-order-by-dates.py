# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# Preview of accident_2016 table
accidents.head("accident_2016")
# info on all columns in the "accident_2016" table
#accidents.table_schema("accident_2016")
# Preview of vehicle_2016 table
accidents.head("vehicle_2016")
# info on all columns in the "vehicles_2016" table
#accidents.table_schema("vehicle_2016")
# query to find out which hour of the day has the most accidents
query1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as HourOfDay, 
                  COUNT(consecutive_number) as Accident_Count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY HourOfDay
            ORDER BY Accident_Count DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query1)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.Accident_Count)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
print(accidents_by_hour)
# query to find out which state has the most hit and runs
query2 = """SELECT registration_state_name as State, 
                  COUNT(hit_and_run) as HitAndRun_Count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY State
            ORDER BY HitAndRun_Count DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hitruns_by_state = accidents.query_to_pandas_safe(query2)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(hitruns_by_state.HitAndRun_Count)
plt.title("Number of Hit And Runs by Rank and State \n (Most to least common)")
print(hitruns_by_state)