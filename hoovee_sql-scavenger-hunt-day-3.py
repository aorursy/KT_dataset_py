# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query1 = """SELECT COUNT(consecutive_number) AS count, EXTRACT(HOUR FROM timestamp_of_crash) AS hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY count DESC
         """

print("Size of query:", accidents.estimate_query_size(query1))
num_of_accidents_per_hour = accidents.query_to_pandas_safe(query1)

num_of_accidents_per_hour
import matplotlib.pyplot as plt
df = num_of_accidents_per_hour.sort_values('hour')
df.plot(x='hour', y='count', kind='bar', title="Number of accidents per hour in 2015")
query2 = """SELECT registration_state_name, COUNT(hit_and_run) as count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            HAVING registration_state_name != "Unknown"
            ORDER BY count DESC
         """

print("Size of query:", accidents.estimate_query_size(query2))
hit_and_runs_in_states = accidents.query_to_pandas_safe(query2)

hit_and_runs_in_states