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
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents1 = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query1 = """SELECT COUNT(consecutive_number), 
                   EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
         """
accidents_by_hour = accidents1.query_to_pandas_safe(query1)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
print(accidents_by_hour)
query2 = """SELECT COUNT(consecutive_number) as The_Amount_Of_Traffic_Accident, 
                   EXTRACT(HOUR FROM timestamp_of_crash) as Every_Hour_In_Day_2015
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Every_Hour_In_Day_2015
            ORDER BY Every_Hour_In_Day_2015
         """
accidents_by_hour_2 = accidents1.query_to_pandas_safe(query2)
print(accidents_by_hour_2)
# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour_2.The_Amount_Of_Traffic_Accident,'red')
plt.axis([0, 23, 500, 2200])
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
plt.xlabel("Every_Hour_In_Day_2015")
plt.ylabel("The_Amount_Of_Traffic_Accident")
query3 = """SELECT COUNT(consecutive_number) as The_Amount_Of_Traffic_Accident, 
                   EXTRACT(HOUR FROM timestamp_of_crash) as Every_Hour_In_Day_2016
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY Every_Hour_In_Day_2016
            ORDER BY Every_Hour_In_Day_2016
         """
accidents_by_hour_3 = accidents1.query_to_pandas_safe(query3)
print(accidents_by_hour_3)
plt.plot(accidents_by_hour_3.The_Amount_Of_Traffic_Accident,'blue')
plt.axis([0, 23, 500, 2200])
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
plt.xlabel("Every_Hour_In_Day_2016")
plt.ylabel("The_Amount_Of_Traffic_Accident")
plt.plot(accidents_by_hour_2.The_Amount_Of_Traffic_Accident,'red',
         accidents_by_hour_3.The_Amount_Of_Traffic_Accident,'blue')
plt.axis([0, 23, 500, 2200])
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
plt.xlabel("Every_Hour_In_Day")
plt.ylabel("The_Amount_Of_Traffic_Accident")
plt.text(1, 2000,'2015', bbox=dict(facecolor='red', alpha=0.5))
plt.text(1, 1800,'2016', bbox=dict(facecolor='blue', alpha=0.5))
query4 = """SELECT COUNT(hit_and_run) AS HitAndRun_Year_2015, 
                   registration_state_name AS State_Name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY State_Name
            ORDER BY HitAndRun_Year_2015
         """
state2015 = accidents1.query_to_pandas_safe(query4)
print(state2015)
query5 = """SELECT COUNT(hit_and_run) AS HitAndRun_Year_2016, 
                   registration_state_name AS State_Name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY State_Name
            ORDER BY HitAndRun_Year_2016
         """
state2016 = accidents1.query_to_pandas_safe(query5)
print(state2016)
accidents_by_hour_2.to_csv("2015_TrafficAccidentAmount_InEveryHour_Use_USTrafficFatalityRecordsDatabase.csv")
accidents_by_hour_3.to_csv("2016_TrafficAccidentAmount_InEveryHour_Use_USTrafficFatalityRecordsDatabase.csv")
state2015.to_csv("2015_HitAndRun_ByState_Use_USTrafficFatalityRecordsDatabase.csv")
state2016.to_csv("2016_HitAndRun_ByState_Use_USTrafficFatalityRecordsDatabase.csv")