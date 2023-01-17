# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head('accident_2016', 1)
query = """SELECT COUNT(consecutive_number) as num_accidents, EXTRACT(hour from timestamp_of_crash) as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hour
            ORDER BY num_accidents"""

accidents.query_to_pandas_safe(query)
accidents.head('vehicle_2016', 1)
accidents.head('vehicle_2016', 1, selected_columns='hit_and_run')
accidents.table_schema("vehicle_2016")
query2 = """SELECT COUNT('drivers_license_state'), state_number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
            GROUP BY state_number
            ORDER BY COUNT('drivers_license_state')"""

accidents.query_to_pandas_safe(query2)
