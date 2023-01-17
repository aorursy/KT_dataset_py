# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.head("accident_2015")

# Which hours of the day do the most accidents occur during?

query_1 = """
    SELECT COUNT(*) AS count, EXTRACT(HOUR FROM timestamp_of_crash) AS hour
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
    GROUP BY hour
    ORDER BY count DESC
"""

accidental_hour = accidents.query_to_pandas_safe(query_1)
accidental_hour
# Which state has the most hit and runs?

query_2 = """
    SELECT COUNT(*) AS hit_and_runs, registration_state_name AS state
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
    WHERE hit_and_run = 'Yes'
    GROUP BY state
    ORDER BY COUNT(*) DESC
"""

hnrs = accidents.query_to_pandas_safe(query_2)
hnrs
