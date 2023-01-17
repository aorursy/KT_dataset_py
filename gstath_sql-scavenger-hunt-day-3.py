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

#* Which hours of the day do the most accidents occur during?
#* Return a table that has information on how many accidents occurred in 
#each hour of the day in 2015, sorted by the the number of accidents which occurred each day.
#Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column.
#(Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to 
#practice with dates. :P)

# query1 = """SELECT COUNT(consecutive_number), 
#                   EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
#             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
#             GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
#             ORDER BY COUNT(consecutive_number) DESC
#         """
# accidents_by_day = accidents.query_to_pandas_safe(query1)

query1 = """SELECT A.count_per_day,B.count_per_hour,B.day,B.hour
            FROM
            (
            SELECT COUNT(consecutive_number) as count_per_day, 
                   EXTRACT(DAYOFWEEK FROM timestamp_of_crash) as day
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
              GROUP BY day
            ) A
            JOIN
            (
            SELECT COUNT(consecutive_number) as count_per_hour, 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) as day,
                  EXTRACT(HOUR FROM timestamp_of_crash) as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY day, hour
            ) B
            ON A.day = B.day
            ORDER BY A.count_per_day DESC,B.count_per_hour,B.hour ASC
        """
accidents_by_day_hour = accidents.query_to_pandas_safe(query1)

query2 =  """SELECT COUNT(consecutive_number) as count_per_hour, 
                  EXTRACT(HOUR FROM timestamp_of_crash) as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY hour
            """
accidents_by_hour = accidents.query_to_pandas_safe(query2)
print(accidents_by_day_hour)
import matplotlib.pyplot as plt
from matplotlib import cm

cmap = cm.get_cmap('Spectral')

fig, axarr = plt.subplots(3, 1, figsize=(12, 10))

accidents_by_day.f0_.plot(ax=axarr[0],colormap=cmap)
accidents_by_hour.plot.bar(x='hour', y='count_per_hour', ax=axarr[1],colormap=cmap)
accidents_by_day_hour.plot.scatter(x='hour', y='count_per_hour', c='day', ax=axarr[2],colormap=cmap)
accidents_by_day_hour.to_csv("accidents_by_day_hour.csv")
#* Which state has the most hit and runs?
#    * Return a table with the number of vehicles registered 
#in each state that were involved in hit-and-run accidents, 
#sorted by the number of hit and runs. 
#Use either the vehicle_2015 or vehicle_2016 table for this, 
#especially the registration_state_name and hit_and_run columns.

query3 = """SELECT registration_state_name, COUNT(hit_and_run) as count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY count DESC
            """

accidents_hitAndRun_by_state = accidents.query_to_pandas_safe(query3)
print(accidents_hitAndRun_by_state)
fig, axarr1 = plt.subplots(1, 1, figsize=(12, 10))

accidents_hitAndRun_by_state.plot.bar(x='registration_state_name', y='count', ax=axarr1,colormap=cmap)
accidents_hitAndRun_by_state.to_csv("accidents_hitAndRun_by_state.csv")