# setup
import bq_helper

fatalities = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                  dataset_name='nhtsa_traffic_fatalities')
query1 = '''SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour, COUNT(*) AS accidents
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        GROUP BY 1
        ORDER BY 2 DESC
        '''

result1 = fatalities.query_to_pandas_safe(query1)
result1
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
ax = sns.barplot(x='hour', y='accidents', data=result1, palette='coolwarm')
query2 = '''SELECT registration_state_name AS state, COUNT(hit_and_run) AS accidents
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        WHERE hit_and_run = 'Yes'
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 20
        '''
result2 = fatalities.query_to_pandas_safe(query2)
result2
ax = plt.subplots(figsize=(15,7))
sns.set_style('darkgrid')
ax = sns.barplot(x='accidents', y='state', data=result2, palette='Blues_r')
