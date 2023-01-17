import bq_helper
import pandas as pd

#object for the dataset
accidents=bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                   dataset_name="nhtsa_traffic_fatalities")
#lets see the list of the tables in the dataset
accidents.list_tables()
accidents.head("accident_2015")
accidents.head("vehicle_2016")
accident_query = """SELECT COUNT(consecutive_number) as Count, 
                  EXTRACT(hour FROM timestamp_of_crash) as Hour_of_day
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour_of_day
            ORDER BY COUNT(consecutive_number) DESC
        """
hourly_accident= accidents.query_to_pandas_safe(query)
hourly_accident
import seaborn as sbn
sbn.barplot(x='Hour_of_day', y='Count', data=hourly_accident).set_title("Accidents by Hour")

hits_query = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
            """
hits= accidents.query_to_pandas_safe(hits_query)
hits
