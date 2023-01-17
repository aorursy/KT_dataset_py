# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query = '''select extract(hour from timestamp_of_crash) as hour, count(consecutive_number) as num_of_acc
           from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           group by hour
           order by num_of_acc desc'''
accidents_by_hour_2016 = accidents.query_to_pandas_safe(query)
accidents_by_hour_2016.head(10)
import matplotlib.pyplot as plt
import numpy as np

plt.plot(accidents_by_hour_2016.num_of_acc)
plt.xticks(np.arange(24),accidents_by_hour_2016.hour)
plt.xlabel("Hour of the day")
plt.ylabel("Number of accidents")
plt.title("Number of accidents by Rank of Hour \n (Most to least dangerous)")

query = '''select registration_state_name, count(hit_and_run) as hit_and_run_count
           from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           where hit_and_run = "Yes" and registration_state_name not in ("No Registration", "Unknown")
           group by registration_state_name
           order by count(hit_and_run) desc'''
accidents.estimate_query_size(query)
hit_and_run_by_state = accidents.query_to_pandas_safe(query)
hit_and_run_by_state.head(10)