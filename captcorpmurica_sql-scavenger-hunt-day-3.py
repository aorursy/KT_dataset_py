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
# Create a visualization on the peak accident hours with sorting by hour.
d3q1a = """SELECT EXTRACT(HOUR FROM timestamp_of_crash),
                  COUNT(consecutive_number) AS freq
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
        """
accidents_by_hour_vis = accidents.query_to_pandas_safe(d3q1a)

plt.plot(accidents_by_hour_vis.freq)
plt.title("Number of Accidents by Hour")
# Your code goes here :)
# Which hours of the day do the most accidents occur?
d3q1 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash),
                 COUNT(consecutive_number) AS freq
          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
          GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
          ORDER BY COUNT(consecutive_number) DESC
       """
accidents_by_hour = accidents.query_to_pandas_safe(d3q1)
print(accidents_by_hour)
# Which state has the most hit and runs?
d3q2 = """SELECT state_name,
                 COUNT(consecutive_number) AS freq
          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
          GROUP BY state_name
          ORDER BY COUNT(consecutive_number) DESC
       """
accidents_by_state = accidents.query_to_pandas_safe(d3q2)
print(accidents_by_state)