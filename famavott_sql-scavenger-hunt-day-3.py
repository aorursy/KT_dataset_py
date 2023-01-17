import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query_1 = """SELECT COUNT(consecutive_number) AS num_of_accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_day
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour_of_day
            ORDER BY num_of_accidents DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query_1)
accidents_by_hour
query_2 = """SELECT registration_state_name AS state,
             COUNT(consecutive_number) as number_hit_and_runs
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             WHERE hit_and_run = 'Yes'
             GROUP BY state
             ORDER BY number_hit_and_runs DESC
          """

hit_and_runs = accidents.query_to_pandas_safe(query_2)
hit_and_runs