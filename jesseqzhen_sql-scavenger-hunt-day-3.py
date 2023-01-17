# import package with helper functions 
import bq_helper
import matplotlib.pyplot as plt
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2015")
query = """
                SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_crash, 
                            COUNT(consecutive_number) AS count
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY hour_of_crash
                ORDER BY count DESC
                """
accidents.estimate_query_size(query)
crash_per_hour = accidents.query_to_pandas_safe(query)
crash_per_hour
crash_sortby_hour = crash_per_hour.sort_values(by='hour_of_crash')
crash_sortby_hour.plot(x='hour_of_crash', y='count');
plt.title("Number of Accidents in Each Hour of the Day in 2015");
accidents.head("vehicle_2015")
query = """
                SELECT registration_state_name, COUNT(hit_and_run) AS count
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                WHERE hit_and_run = 'Yes'
                GROUP BY registration_state_name
                ORDER BY count DESC
"""
accidents.estimate_query_size(query)
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
hit_and_run_by_state