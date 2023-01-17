# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# Listing tables
accidents.list_tables()
# Answer to Question 1

query_1 = """
                SELECT COUNT(consecutive_number) AS No_of_Accidents,
                EXTRACT(HOUR FROM timestamp_of_crash) AS Time_of_Crash
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY Time_of_Crash
                ORDER BY No_of_Accidents DESC
         """

worst_hours = accidents.query_to_pandas_safe(query_1)

worst_hours.head()
# Answer to Question 2

query_2 = """
                SELECT registration_state_name, hit_and_run, COUNT(hit_and_run) AS No_of_hit_and_Run
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                GROUP BY registration_state_name, hit_and_run
                HAVING hit_and_run = 'Yes'
                ORDER BY COUNT(hit_and_run) DESC
         """

worst_state = accidents.query_to_pandas_safe(query_2)

worst_state.head()

