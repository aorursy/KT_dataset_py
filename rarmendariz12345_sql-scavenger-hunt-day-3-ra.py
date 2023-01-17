# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
acc_hour_query = """SELECT COUNT(consecutive_number) as Number_of_Accidents, EXTRACT(HOUR FROM timestamp_of_crash)as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY hour 
        """
#Create pandas query table
acc_hour = accidents.query_to_pandas_safe(acc_hour_query)
acc_hour
#Query statement for hit and run
hitrun_query = """SELECT registration_state_name as state, COUNT(hit_and_run) as number_of_hit_and_runs
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                WHERE hit_and_run = 'Yes'
                GROUP BY state
                ORDER BY COUNT(hit_and_run) desc"""

#Create pandas query table
hitrun = accidents.query_to_pandas_safe(hitrun_query)
hitrun