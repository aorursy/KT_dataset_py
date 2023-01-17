# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# Your Code Here
query = """SELECT COUNT(consecutive_number) as amount, 
            EXTRACT(DAYOFWEEK FROM timestamp_of_crash) as day
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY day
            ORDER BY amount DESC
        """

accidents_by_day = accidents.query_to_pandas_safe(query, max_gb_scanned=0.1)
print(accidents_by_day)
# Your Code Here
accidents.table_schema("vehicle_2016")
accidents.head("vehicle_2016")
query = """SELECT COUNT(consecutive_number) as amount, 
            registration_state_name as state
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY state
            ORDER BY amount DESC
        """

state_hr = accidents.query_to_pandas_safe(query, max_gb_scanned=0.1)
print(state_hr)