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
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
accidents.head('accident_2016')
# Your code goes here :)

query1 = """
            SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
hour_accident = accidents.query_to_pandas_safe(query1)

hour_accident
hour_accident.to_csv('hour_accident.csv')
import seaborn as sns

sns.set_style("whitegrid") 
ax = sns.barplot(x="f1_", y="f0_", data=hour_accident, palette="coolwarm")

accidents.head('vehicle_2016')
query2 = """
            SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
hit_and_run_states = accidents.query_to_pandas_safe(query2)

hit_and_run_states
hit_and_run_states.to_csv('hit_and_run_states.csv')
ax = plt.subplots(figsize=(6, 18))
sns.set_style("whitegrid") 
ax = sns.barplot(x="f0_", y="registration_state_name", data=hit_and_run_states, palette="coolwarm")

