# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.table_schema('accident_2015')
accidents.head('accident_2015').columns
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
# 1. During which hours of the day do the most accidents occur?
#    a. sort by number of accidents which occurred each hour.

query = """SELECT COUNT(consecutive_number) as crashes,
           EXTRACT(HOUR FROM timestamp_of_crash) AS hour
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY hour
           ORDER BY crashes DESC
        """

hourly_accident = accidents.query_to_pandas_safe(query)

hourly_accident
# 2. Which state has the most hit-and-runs?
accidents.table_schema('vehicle_2016')
accidents.head('vehicle_2016')['hit_and_run']
query = """SELECT registration_state_name, 
           COUNT(registration_state_name) as num_hit_and_run
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
           WHERE hit_and_run = 'Yes'
           GROUP BY registration_state_name
           ORDER BY num_hit_and_run DESC
        """

hit_and_runs = accidents.query_to_pandas_safe(query)
hit_and_runs