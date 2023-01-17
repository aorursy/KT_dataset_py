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
# Your code goes here :)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to find out the number of accidents which 
# happen on each hour
query = """SELECT COUNT(consecutive_number) counter, 
                  EXTRACT(HOUR FROM timestamp_of_crash) hours
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hours
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query)
# library for plotting
import seaborn as sns

# make a plot to show that our data is, actually, sorted:
# set_style - darkgrid, whitegrid, dark, white, ticks
f, ax = plt.subplots(figsize=(10, 7))
sns.set_style("whitegrid") 
ax = sns.barplot(x="hours", y="counter", data=accidents_by_hour,palette='Blues')

accidents_by_hour
# query to find out the state has the most hit and runs
query = """SELECT registration_state_name state,
                  COUNT(consecutive_number) counter
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
           WHERE  hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hit_and_run = accidents.query_to_pandas_safe(query)
# library for plotting
import seaborn as sns

# set_style - darkgrid, whitegrid, dark, white, ticks
f, ax = plt.subplots(figsize=(7, 20))
sns.set_style("whitegrid") 
ax = sns.barplot(x="counter", y="state", data=accidents_by_hit_and_run,palette='Set2',dodge=False)

accidents_by_hit_and_run