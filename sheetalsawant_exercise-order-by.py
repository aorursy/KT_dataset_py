# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.table_schema("accident_2015")
# Your Code Here
query = """SELECT COUNT(consecutive_number) AS number_of_accidents,
           EXTRACT (HOUR FROM timestamp_of_crash) AS hour_of_the_day
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY hour_of_the_day
           ORDER BY number_of_accidents DESC
        """

accident_per_hour = accidents.query_to_pandas(query)
accident_per_hour


import matplotlib.pyplot as plt
plt.plot(accident_per_hour.number_of_accidents,accident_per_hour.hour_of_the_day)
plt.title("Accidents according to hour of the day")
import matplotlib.pyplot as plt
plt.bar(accident_per_hour.hour_of_the_day,accident_per_hour.number_of_accidents)
plt.title("Accidents according to hour of the day")
accidents.table_schema("vehicle_2015")
accidents.head("vehicle_2015")
# Your Code Here
query = """SELECT COUNTIF(hit_and_run='Yes') AS hit_and_run_number,
           registration_state_name
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           GROUP BY registration_state_name
           ORDER BY hit_and_run_number DESC
        """

hitandrun = accidents.query_to_pandas(query)
hitandrun