# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2015")
# define query:

query1 = """SELECT COUNT(consecutive_number), 
            EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC

         """

# submit query using the safe, scan-size limited function
fatalitiesByHour = accidents.query_to_pandas_safe(query1)

# print the result
fatalitiesByHour
# define query:

query2 = """SELECT registration_state_name, COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
         """

# submit query using the safe, scan-size limited function
hitRun = accidents.query_to_pandas_safe(query2)

# print the result
hitRun