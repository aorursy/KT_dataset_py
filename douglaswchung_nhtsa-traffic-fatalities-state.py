# import package with helper functions 

import bq_helper



# create a helper object for this dataset

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="nhtsa_traffic_fatalities")
accidents.table_schema("accident_2016")
# Your Code Here

query = """SELECT COUNT(consecutive_number) AS num_accidents, 

                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_day

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY hour_of_day

            ORDER BY num_accidents DESC

        """

accidents_by_hour = accidents.query_to_pandas_safe(query)

print(accidents_by_hour)
# Your Code Here

query = """ SELECT registration_state_name, COUNT(consecutive_number) AS num_accidents

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`

            WHERE hit_and_run = "Yes"

            GROUP BY registration_state_name

            ORDER BY num_accidents DESC

"""

hit_and_runs_per_state = accidents.query_to_pandas_safe(query)

print(hit_and_runs_per_state)