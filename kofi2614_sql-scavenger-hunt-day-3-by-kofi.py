# Which hours of the day do the most accidents occur during?
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query_1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(Hour FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOur FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query_1)
print(accidents_by_hour)
import matplotlib.pyplot as plt
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hours \n (Most to least dangerous)")
# Which state has the most hit and runs?
query_2 = """SELECT COUNT(hit_and_run),registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
          """

accidents_by_state = accidents.query_to_pandas_safe(query_2)
print(accidents_by_state)