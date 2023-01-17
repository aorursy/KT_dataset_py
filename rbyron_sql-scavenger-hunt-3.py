# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head('accident_2016')
# query to find out the number of accidents which 
# happen on each hour of the day
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(15, 5))
sns.barplot(x='f1_', y='f0_', data=accidents_by_hour)
plt.xlabel('Hour of the day')
plt.ylabel('No. accidents')
plt.title("No. accidents by hour of the day")
accidents_by_hour.head(5)
accidents.head('vehicle_2016').columns.values
accidents.head('vehicle_2016', 5).loc[:, ['registration_state_name', 'hit_and_run']]
# query to find out the number of accidents which 
# happen on each hour of the day
query = """ SELECT registration_state_name, COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_and_runs = accidents.query_to_pandas_safe(query)
hit_and_runs.head(11)
plt.figure(figsize=(15, 5))
plotdata = hit_and_runs.head(11)[1:]
pal = np.flip(np.array(sns.color_palette("coolwarm", len(plotdata))), 0)

sns.barplot(x='registration_state_name', y='f0_', data=plotdata, palette=pal)
plt.xlabel('State')
plt.ylabel('No. hit-and-runs')
