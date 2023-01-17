# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# Your Code Here
query31 = """SELECT COUNT(consecutive_number),
                    EXTRACT(HOUR FROM timestamp_of_crash)
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                ORDER BY COUNT(consecutive_number) DESC"""
accidents_by_hour = accidents.query_to_pandas_safe(query31)
print(accidents_by_hour)
# Your Code Here
query32 = """SELECT registration_state_name, COUNTIF(hit_and_run = "Yes")
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                 GROUP BY registration_state_name
                 ORDER BY COUNTIF(hit_and_run = "Yes") DESC
                """
accidents.query_to_pandas_safe(query32)