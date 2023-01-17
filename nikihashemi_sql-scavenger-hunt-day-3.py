#Part 1: Which hours of the day do the most accidents occur?
import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash), COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)
accidents_by_hour
#Part 2: Which state has the most hit and runs
import bq_helper

vehicles = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query2 = """SELECT registration_state_name,COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run)
        """
hit_and_run_state = accidents.query_to_pandas_safe(query2)
hit_and_run_state