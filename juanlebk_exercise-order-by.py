# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2015")
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour, 
            COUNT(consecutive_number) AS accident_count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY accident_count DESC"""
accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour
import seaborn as sns
sns.barplot(x='hour', y='accident_count', data=accidents_by_hour)
accidents.head('vehicle_2015')
query = """SELECT registration_state_name AS state, COUNTIF(hit_and_run = "Yes") AS vehicle_count_hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY state
            ORDER BY vehicle_count_hit_and_run"""
num_vehicle_hit_and_run_by_state = accidents.query_to_pandas_safe(query)
num_vehicle_hit_and_run_by_state.sort_values(by='vehicle_count_hit_and_run', ascending= False)