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
query1 = """SELECT COUNT(consecutive_number) as `total`, 
                  EXTRACT(HOUR FROM timestamp_of_crash) as `hour_of_crash`
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY `hour_of_crash`
            ORDER BY `hour_of_crash`
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)
plt.plot(accidents_by_hour.hour_of_crash, accidents_by_hour.total)
plt.title("Number of Accidents in Hour \n (Most to least dangerous)")
query2 = """SELECT `registration_state_name`, COUNT(`hit_and_run`) as `hit_and_runs`
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE `hit_and_run` = 'Yes' AND `registration_state_name`!= 'Unknown'
            GROUP BY `registration_state_name`
            ORDER BY `hit_and_runs`
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(query2)
plt.subplots(figsize=(10, 10))
plt.barh(hit_and_run_by_state.registration_state_name, hit_and_run_by_state.hit_and_runs)
plt.title("Number of Accidents in Hour \n (Most to least dangerous)")