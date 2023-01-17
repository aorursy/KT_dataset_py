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

import bq_helper
big_query = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="nhtsa_traffic_fatalities")

# Which hours of the day do the most accidents occur during?
query = """
        select extract(hour from timestamp_of_crash)
        from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        group by extract(hour from timestamp_of_crash)
        order by count(*)
        limit 1
        """

accidents_by_hour = big_query.query_to_pandas_safe(query)
accidents_by_hour.head()
# Which state has the most hit and runs?
query = """
        select registration_state_name, count(hit_and_run)
        from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        where hit_and_run = "Yes"
        group by registration_state_name
        order by count(hit_and_run) desc
        """

hit_and_run_by_state = big_query.query_to_pandas_safe(query)
hit_and_run_by_state.head()