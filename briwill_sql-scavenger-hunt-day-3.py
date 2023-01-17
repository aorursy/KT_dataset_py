# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number) as crashes, 
                  EXTRACT(HOUR FROM timestamp_of_crash) as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY crashes DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query)

accidents_by_hour
query = """SELECT registration_state_name, COUNT(vehicle_number) as vehicles
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY vehicles DESC
        """

vehicles_by_state = accidents.query_to_pandas_safe(query)

vehicles_by_state