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
# Question 1: Which hours of the day do the most accidents occur during?
#    * Return a table that has information on how many accidents occurred in each hour of the day in 2015,
#      sorted by the the number of accidents which occurred each hour. Use the accident_2015 or 
#      accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an 
#      hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
#    * **Hint:** You will probably want to use the [HOUR() function]

# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query)

print(accidents_by_hour)

# library for plotting
import matplotlib.pyplot as plt

# first, plot from accidents by hour to see that they're sorted correctly\
plt.figure(0)
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")

# now, as far as plotting, to me it's preferable to see the accidents *not sorted* by hour
# but rather, unsorted so you can see accident trends throughout a day
plt.figure(1)
unsorted_hour_accidents = accidents_by_hour.sort_values( by = ['f1_'], axis=0, ascending=True)
plt.plot(unsorted_hour_accidents.f1_, unsorted_hour_accidents.f0_)
plt.title("Number of Accidents by Hour of Day")
# Question 1: * Which state has the most hit and runs?
#    * Return a table with the number of vehicles registered in each state that were involved in 
#      hit-and-run accidents, sorted by the number of hi Use the vehicle_2015 or vehicle_2016 table 
#      for this, especially the registration_state_name and hit_and_run columns.

# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_and_runs = accidents.query_to_pandas_safe(query)
print(hit_and_runs)
import seaborn as sns
import matplotlib.pyplot as plt

trimmed_data = hit_and_runs[1:]
fig, ax = plt.subplots(figsize=(10,7))
bp = sns.barplot( x="registration_state_name", y="f0_", data=trimmed_data)
for item in bp.get_xticklabels():
    item.set_rotation(90)