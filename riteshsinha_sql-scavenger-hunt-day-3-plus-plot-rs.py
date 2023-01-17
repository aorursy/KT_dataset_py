

# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query_accidents_by_hour=""" SELECT COUNT (consecutive_number),
                EXTRACT(HOUR FROM timestamp_of_crash)
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                GROUP BY EXTRACT (HOUR FROM timestamp_of_crash)
                ORDER BY COUNT(consecutive_number) DESC
        """
df_accidents_by_hour= accidents.query_to_pandas(query_accidents_by_hour)

df_accidents_by_hour.head()

import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(df_accidents_by_hour.f1_, df_accidents_by_hour.f0_)
plt.title("Number of Accidents by Hour of Day \n (Most to least dangerous)")
query_accidents_by_state = """SELECT registration_state_name, COUNT(*) as count
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        WHERE hit_and_run = "Yes"
        GROUP BY registration_state_name
        ORDER BY count DESC
    """
        
df_query_accidents_by_state = accidents.query_to_pandas_safe(query_accidents_by_state)
df_query_accidents_by_state.head()
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(14,10))
ax = sns.barplot(x = 'registration_state_name', y ='count', data = df_query_accidents_by_state)
for item in ax.get_xticklabels():
    item.set_rotation(45)