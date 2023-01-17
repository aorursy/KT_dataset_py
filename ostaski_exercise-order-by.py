# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query = """SELECT COUNT(consecutive_number) AS Accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS Hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour
            ORDER BY COUNT(consecutive_number) DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query)
print(accidents_by_hour)
query = """SELECT COUNT(hit_and_run) AS HitAndRuns, 
                  registration_state_name AS State
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """

hit_runs_by_state = accidents.query_to_pandas_safe(query)
print(hit_runs_by_state)