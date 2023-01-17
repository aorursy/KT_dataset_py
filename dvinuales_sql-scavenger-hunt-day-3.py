# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as Hour, 
            COUNT(consecutive_number) as Count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE 
                EXTRACT(YEAR FROM timestamp_of_crash) = 2015
            GROUP BY Hour
            ORDER BY Count DESC
        """
hourly_accidents = accidents.query_to_pandas_safe(query)

hourly_accidents.head(hourly_accidents.shape[0])
query = """SELECT registration_state_name as State, 
            COUNTIF(hit_and_run = 'Yes') as Count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`            
            GROUP BY State
            HAVING Count > 0
            ORDER BY Count DESC
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(query)

hit_and_run_by_state.head(hit_and_run_by_state.shape[0])