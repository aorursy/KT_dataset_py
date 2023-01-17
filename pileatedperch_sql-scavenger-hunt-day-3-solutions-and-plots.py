import bq_helper
nhtsa_traffic_fatalities = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                                    dataset_name = 'nhtsa_traffic_fatalities')
nhtsa_traffic_fatalities.head('accident_2015')
accidents_by_day = nhtsa_traffic_fatalities.query_to_pandas_safe("""
SELECT
    EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week,
    COUNT(consecutive_number) AS num_accidents
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY day_of_week
ORDER BY num_accidents DESC
""")

accidents_by_day
day_dict = {1:'Sun', 2:'Mon', 3:'Tue', 4:'Wed', 5:'Thu', 6:'Fri', 7:'Sat'}
accidents_by_day['day_of_week'] = accidents_by_day['day_of_week'].map(day_dict)
accidents_by_day
import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x='day_of_week', y='num_accidents', data=accidents_by_day, order=day_dict.values(), color='darkblue')
plt.title('Number of Accidents by Day of the Week in 2015')
plt.ylabel('Numer of Accidents')
plt.xlabel('Day of the Week')
accidents_by_hour = nhtsa_traffic_fatalities.query_to_pandas_safe("""
SELECT
    EXTRACT(HOUR FROM timestamp_of_crash) AS hour,
    COUNT(consecutive_number) AS num_accidents
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY hour
ORDER BY num_accidents DESC
""")

accidents_by_hour
plt.figure(figsize=(10,6))
sns.barplot(y='num_accidents', x='hour', data=accidents_by_hour, color='darkblue')
plt.title('Number of Accidents by Hour of the Day in 2015')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Accidents')
nhtsa_traffic_fatalities.head('vehicle_2015')
nhtsa_traffic_fatalities.query_to_pandas_safe("""
SELECT DISTINCT hit_and_run
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
""")
num_hit_and_runs = nhtsa_traffic_fatalities.query_to_pandas_safe("""
SELECT registration_state_name AS state, COUNT(consecutive_number) AS num_hit_and_runs
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
WHERE hit_and_run = 'Yes'
GROUP BY state
ORDER BY num_hit_and_runs DESC
""")
num_hit_and_runs
plt.figure(figsize=(6,15))
sns.barplot(x='num_hit_and_runs', y='state', data=num_hit_and_runs, orient='h', color='darkblue')
plt.title('Number of Hit-and-runs by State in 2015')
plt.xlabel('Number of hit-and-runs')
plt.ylabel('State')