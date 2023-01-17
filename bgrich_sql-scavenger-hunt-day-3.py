# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.table_schema("accident_2015")
query_crashes_per_hour = """SELECT EXTRACT(HOUR FROM timestamp_of_crash), 
                                   COUNT(consecutive_number) 
                            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
                         GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                         ORDER BY 1
                      """
accidents.estimate_query_size(query_crashes_per_hour) * 1000
crashes_per_hour = accidents.query_to_pandas_safe(query_crashes_per_hour)
print(crashes_per_hour)
accidents.table_schema("vehicle_2015")
query_most_hit_run = """SELECT state_number, COUNT(consecutive_number)
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
                        WHERE hit_and_run = 'Yes'
                        GROUP BY state_number
                        ORDER BY COUNT(consecutive_number) DESC"""
accidents.estimate_query_size(query_most_hit_run)
most_hit_run = accidents.query_to_pandas_safe(query_most_hit_run)
print(most_hit_run)