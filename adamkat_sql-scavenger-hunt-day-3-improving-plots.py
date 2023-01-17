# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.bar(accidents_by_day.f1_,accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
plt.bar(accidents_by_day.f1_,accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
query1 = """SELECT COUNT(consecutive_number) number_of_accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) hour_of_accident
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour_of_accident
            ORDER BY number_of_accidents DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)
import numpy as np
plt.bar(accidents_by_hour.hour_of_accident,accidents_by_hour.number_of_accidents)
plt.xticks(np.arange(0,24,2))
plt.title("Number of Accidents by hour of day")
plt.show()
query2 = """SELECT registration_state_name state,
                   COUNT(consecutive_number) number_of_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           
            GROUP BY state
            ORDER BY number_of_accidents ASC
            LIMIT 20
        """
hit_n_run_by_state = accidents.query_to_pandas_safe(query2)
fig, ax = plt.subplots(figsize=(15,8))

y_pos = np.arange(len(hit_n_run_by_state.number_of_accidents))
ax.barh(y_pos,hit_n_run_by_state.number_of_accidents,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(hit_n_run_by_state.state)
ax.set_title("Number of Accidents by state")
plt.show()