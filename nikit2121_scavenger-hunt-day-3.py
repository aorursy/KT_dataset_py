import pandas as pd 
from bq_helper import BigQueryHelper
import seaborn as sns
from matplotlib import pyplot as plt
bq_US_accidents_2015 =BigQueryHelper(active_project='bigquery-public-data',dataset_name='nhtsa_traffic_fatalities')
bq_US_accidents_2015.head('accident_2015')
query = """Select EXTRACT(Hour from timestamp_of_crash), COUNT(consecutive_number) as total
        from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        group by EXTRACT(Hour from timestamp_of_crash)
        order by COUNT(consecutive_number) DESC
        """
bq_US_accidents_2015.estimate_query_size(query)
accidents_by_hours = bq_US_accidents_2015.query_to_pandas(query)
accidents_by_hours
plt.figure(figsize=(10,8))
sns.barplot(x='f0_',y='total',data=accidents_by_hours,color='blue')
plt.show()
bq_US_accidents_2015.head('vehicle_2015')
query = """SELECT COUNT(consecutive_number) as count, registration_state_name
            from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run="Yes"
            group by registration_state_name
            order by count(consecutive_number) DESC
            
        
        """
bq_US_accidents_2015.estimate_query_size(query)
hit_and_run = bq_US_accidents_2015.query_to_pandas(query)
top_5_hit_and_run_states = hit_and_run.head(11)
top_5_hit_and_run_states
plt.figure(figsize=(13,8))
sns.barplot(x='registration_state_name',y='count',data=top_5_hit_and_run_states)
plt.title('TOP 10 State with higher rate of accidents')
plt.show()