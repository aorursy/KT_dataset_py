# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
#Which hours of the day do the most accidents occur during?
query1=""" SELECT COUNT (consecutive_number),
                EXTRACT(HOUR FROM timestamp_of_crash)
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                GROUP BY EXTRACT (HOUR FROM timestamp_of_crash)
                ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour= accidents.query_to_pandas(query1)

accidents_by_hour.head()
#Which state has the most hit and runs?

query2 = """SELECT registration_state_name, COUNT(*) as count
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        WHERE hit_and_run = "Yes"
        GROUP BY registration_state_name
        ORDER BY count DESC
    """
        
hit_run_state=accidents.query_to_pandas_safe(query2)
hit_run_state.head()