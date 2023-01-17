import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash), COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query1)

query2 = """SELECT registration_state_name, count(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """

hitandruns_by_state = accidents.query_to_pandas_safe(query2)

print('Q1\n',accidents_by_hour,'\nQ2\n',hitandruns_by_state)