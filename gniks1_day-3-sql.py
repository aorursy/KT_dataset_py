import bq_helper
Traffic_Data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
dataset_name="nhtsa_traffic_fatalities")
d3_q1 = """SELECT COUNT(consecutive_number) AS total_crashes,
                     EXTRACT(HOUR FROM timestamp_of_crash) AS Time_of_Crash
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
              GROUP BY Time_of_Crash
              ORDER BY total_crashes"""
crash_table = Traffic_Data.query_to_pandas_safe(d3_q1)
print(crash_table)
d3_q2 = """SELECT registration_state_name,
                     COUNTIF(hit_and_run = 'Yes') AS Positive_Cases
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
              GROUP BY registration_state_name
              ORDER BY Positive_Cases"""
Hit_Run_Table = Traffic_Data.query_to_pandas_safe(d3_q2)
print(Hit_Run_Table)