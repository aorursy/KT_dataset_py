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

# Scavenger hunt
# Which hours of the day do the most accidents occur during?

query1 ="""SELECT COUNT(consecutive_number) as number_of_accident_hour, 
           EXTRACT(HOUR FROM timestamp_of_crash) as time_in_hr 
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
           GROUP BY time_in_hr 
           ORDER BY number_of_accident_hour DESC """

accidents_by_hour=accidents.query_to_pandas_safe(query1)
accidents_by_hour

# Scavenger Hunt
# Which state has the most hit and runs?
query2 ="""SELECT registration_state_name as stateName,hit_and_run,
           COUNT(hit_and_run) as count_hit_and_run 
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` 
           GROUP BY stateName,hit_and_run 
           HAVING hit_and_run='Yes' 
           ORDER BY count_hit_and_run DESC """

accidents.query_to_pandas_safe(query2)
