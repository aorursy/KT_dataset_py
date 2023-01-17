import bq_helper
accident = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                        dataset_name = 'nhtsa_traffic_fatalities')
accident.list_tables()
accident.table_schema('accident_2016')
query = """SELECT COUNT(consecutive_number), EXTRACT (HOUR FROM timestamp_of_crash)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        GROUP BY EXTRACT (HOUR FROM timestamp_of_crash)
        ORDER BY COUNT(consecutive_number) DESC
"""
accident.estimate_query_size(query)

df_accidents1 = accident.query_to_pandas_safe(query)
print (df_accidents1)
print (df_accidents1['f0_'].sum())
query1 = """SELECT COUNT(consecutive_number)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        """
df_accidents2 = accident.query_to_pandas_safe(query1)
print (df_accidents2)
accident.table_schema('vehicle_2016')
query2 = """SELECT registration_state_name, COUNT(hit_and_run)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        WHERE hit_and_run = 'Yes'
        GROUP BY registration_state_name
        ORDER BY COUNT(hit_and_run)
        """
accident.estimate_query_size(query2)
df_accident3 = accident.query_to_pandas_safe(query2)
print (df_accident3)
query3 = """SELECT registration_state_name, COUNT(hit_and_run)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        WHERE hit_and_run = 'Yes'
        GROUP BY registration_state_name
        ORDER BY registration_state_name
        """
accident.estimate_query_size(query3)
df_accident4 = accident.query_to_pandas_safe(query3)
print (df_accident4)