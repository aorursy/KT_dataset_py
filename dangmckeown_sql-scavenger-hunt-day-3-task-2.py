# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

#accidents.head("damage_2016")

query = """ SELECT registration_state_name, COUNT(hit_and_run) AS number_hit_and_runs
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
        WHERE hit_and_run = 'Yes'
        GROUP BY registration_state_name
        ORDER BY COUNT(hit_and_run) DESC"""

prangs = accidents.query_to_pandas_safe(query)
#accidents.head("vehicle_2015")
prangs