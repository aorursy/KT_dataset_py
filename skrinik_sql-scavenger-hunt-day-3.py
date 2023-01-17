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
accidents_by_day
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Your code goes # import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
#Which hours of the day do the most accidents occur during?
    #Return a table that has information on how many accidents occurred in each hour 
    #of the day in 2015, sorted by the the number of accidents which occurred each day. 
    #Use the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. 
query1 = """
select count(consecutive_number	) as num_accidents, 
extract(hour from timestamp_of_crash) as hour_crash
from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
group by hour_crash
order by num_accidents desc
"""
accidents_hourly = accidents.query_to_pandas_safe(query1)
print(accidents_hourly)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_hourly.hour_crash,accidents_hourly.num_accidents, 'r.')
plt.title("Number of Accidents by Hour")
#Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state 
#that were involved in hit-and-run accidents, sorted by the number 
#of hi Use the vehicle_2015 or vehicle_2016 table for this, 
#especially the registration_state_name and hit_and_run columns.

query2 = """
select count(consecutive_number) as num_hit_run, state_number
from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
group by state_number
order by num_hit_run desc
"""
accidents_state = accidents.query_to_pandas_safe(query2)
print(accidents_state)

# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_state.state_number,accidents_state.num_hit_run, 'b.')
plt.title("Number of Accidents by State")