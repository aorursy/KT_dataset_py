import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.head("accident_2015")
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
print(accidents_by_hour)
import matplotlib.pyplot as plt
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
accidents.head("vehicle_2015")
query = """SELECT state_number, consecutive_number, hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            LIMIT 10 
        """
temp = accidents.query_to_pandas_safe(query)
print(temp)
query = """SELECT COUNT(consecutive_number), state_number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY state_number
            ORDER BY COUNT(consecutive_number) DESC
        """
hitandrun = accidents.query_to_pandas_safe(query)
print(hitandrun)
query = """SELECT state_number, consecutive_number, hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE state_number = 6
            AND hit_and_run = "Yes"
        """
temp = accidents.query_to_pandas_safe(query)
print(temp.shape)