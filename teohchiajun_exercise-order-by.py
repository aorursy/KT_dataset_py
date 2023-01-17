# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2016")
# query to find out the number of accidents which happen on each hours of the day
query = """SELECT COUNT(consecutive_number),
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hours = accidents.query_to_pandas_safe(query)
print(accidents_by_hours)
# Your Code Here
query_2 = """SELECT registration_state_name, COUNT(hit_and_run)
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
              GROUP BY registration_state_name
              ORDER BY COUNT(hit_and_run) DESC
          """
hit_and_run_accidents = accidents.query_to_pandas_safe(query_2)
print(hit_and_run_accidents)