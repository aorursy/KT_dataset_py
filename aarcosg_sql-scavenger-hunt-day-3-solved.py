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
import seaborn as sns

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# query to find out the number of accidents which 
# happen on each hour of the day
query = """SELECT COUNT(consecutive_number),
               EXTRACT(HOUR FROM timestamp_of_crash)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
           ORDER BY COUNT(consecutive_number) DESC
        """
accidents_per_hour = accidents.query_to_pandas_safe(query)
accidents_per_hour.head()
# plot accidents per hour
plt.bar(accidents_per_hour.f1_,accidents_per_hour.f0_)
plt.title('Number of Accidents by Rank of Hour')
print(accidents_per_hour)
# check table vehicle_2015
accidents.head('vehicle_2015')
# query to find out which state has the most hit and runs
query = """SELECT registration_state_name AS state, COUNT(consecutive_number) AS num_hit_and_runs
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           WHERE hit_and_run = 'Yes'
           GROUP BY registration_state_name
           ORDER BY COUNT(consecutive_number) DESC
        """
hit_and_runs_per_state = accidents.query_to_pandas_safe(query)
hit_and_runs_per_state.head()
#plot results
f, ax = plt.subplots(figsize=(7, 14))
sns.barplot(x="num_hit_and_runs", y="state", data=hit_and_runs_per_state)
print(hit_and_runs_per_state)