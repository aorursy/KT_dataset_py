import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                     dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2015")
query = """SELECT COUNT(consecutive_number) AS num_accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`       
           GROUP BY hour
           ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour
accidents_by_hour.plot(x='hour',y='num_accidents',kind='bar')
accidents_by_hour.sort_values(by='hour').plot(x='hour',y='num_accidents',kind='bar')
accidents.query_to_pandas("""SELECT DISTINCT hit_and_run
                             FROM`bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`""")
accidents.query_to_pandas("""SELECT DISTINCT registration_state_name
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`""")
query = """SELECT registration_state_name,
                  COUNT(consecutive_number) AS hit_and_run_accidents
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           WHERE hit_and_run = 'Yes'
           GROUP BY registration_state_name
           ORDER BY COUNT(consecutive_number) DESC
        """
hit_and_run = accidents.query_to_pandas_safe(query)
hit_and_run