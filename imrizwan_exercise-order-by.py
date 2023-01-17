# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.list_tables()
accidents.head('accident_2015')
accidents.head('accident_2016')
query_1 = """SELECT consecutive_number, 
                    EXTRACT(HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             ORDER BY consecutive_number DESC
             """
accidents_by_hour = accidents.query_to_pandas_safe(query_1)
print(accidents_by_hour)
query_2 = """SELECT COUNT(consecutive_number), 
                    EXTRACT(HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
             """
accidents_by_hour = accidents.query_to_pandas_safe(query_2)
print(accidents_by_hour)


accidents.head("vehicle_2015")
accidents.head("vehicle_2016")
query_3 = """SELECT registration_state_name, COUNT(hit_and_run)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             GROUP BY registration_state_name
             ORDER BY COUNT(hit_and_run)
             """
accidents_by_hour = accidents.query_to_pandas_safe(query_3)
print(accidents_by_hour)