import bq_helper
accidents = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                     dataset_name='nhtsa_traffic_fatalities')
accidents.list_tables()
sorted(accidents.head('accident_2015').columns)
query = '''SELECT COUNT(consecutive_number) AS num_accidents,
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS weekday
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY weekday
           ORDER BY num_accidents DESC'''
accidents.estimate_query_size(query)
num_accidents_by_weekday = accidents.query_to_pandas_safe(query)
num_accidents_by_weekday
num_accidents_by_weekday['weekday'] = num_accidents_by_weekday['weekday'].map({1:'Sun', 2:'Mon', 3:'Tue', 4:'Wed', 5:'Thu', 6:'Fri', 7:'Sat'})
num_accidents_by_weekday
num_accidents_by_weekday.plot(x='weekday', y='num_accidents', kind='bar')
query = '''SELECT COUNT(consecutive_number) AS num_accidents,
           EXTRACT(HOUR FROM timestamp_of_crash) AS hour
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY hour
           ORDER BY num_accidents DESC'''
accidents.estimate_query_size(query)
num_accidents_by_hour = accidents.query_to_pandas_safe(query)
num_accidents_by_hour
num_accidents_by_hour.plot(x='hour', y='num_accidents', kind='bar')
sorted(accidents.head('vehicle_2015').columns)
query = '''SELECT hit_and_run, COUNT(*) AS num_values
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           GROUP BY hit_and_run'''
accidents.estimate_query_size(query)
hit_and_run = accidents.query_to_pandas_safe(query)
hit_and_run
query = '''SELECT state_number, COUNT(*) AS num_hit_and_runs
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           WHERE hit_and_run = 'Yes'
           GROUP BY state_number
           HAVING num_hit_and_runs > 30
           ORDER BY num_hit_and_runs DESC'''
accidents.estimate_query_size(query)
num_hit_and_runs_by_state = accidents.query_to_pandas_safe(query)
num_hit_and_runs_by_state
num_hit_and_runs_by_state.plot(x='state_number', y='num_hit_and_runs', kind='bar')