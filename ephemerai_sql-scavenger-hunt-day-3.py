# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head("accident_2015")
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

# ggplot
from matplotlib import style
style.use("ggplot")

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Your code goes here :)
from bq_helper import BigQueryHelper
accident = BigQueryHelper(active_project = "bigquery-public-data",
                          dataset_name = "nhtsa_traffic_fatalities")
# Which hours of the day do the most accidents occur during?
accident.head("accident_2016")

query1 = """
            SELECT COUNT(consecutive_number),
                    EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# estimate the size of query1
accident.estimate_query_size(query1)
# store in a data frame
accident_by_hour = accident.query_to_pandas_safe(query1)
accident_by_hour.head()
# plot accident_by_hour
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

plt.plot(accident_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
# Which state has the most hit and runs?
accident.head("vehicle_2016")

query2 = """
            SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
# estimate the size of query2
accident.estimate_query_size(query2)
# store in a data frame
state_hit_and_run = accident.query_to_pandas_safe(query2)
state_hit_and_run.head()
# plot accident_by_hour
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

plt.plot(state_hit_and_run.f0_)
plt.title("Number of Hit and Runs by State \n (Most to least dangerous)")