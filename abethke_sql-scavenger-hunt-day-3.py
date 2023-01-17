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
# Hour Frequency
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY 1 DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
print(accidents_by_hour)
query = """SELECT COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run="Yes"
        """
state_hrs = accidents.query_to_pandas_safe(query)
print(state_hrs)
# Hit and runs
query = """SELECT registration_state, COUNT(consecutive_number) as hit_and_runs
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run="Yes" 
            GROUP BY registration_state
            ORDER By 2 DESC
        """
state_hrs = accidents.query_to_pandas_safe(query)
print(state_hrs.head(10))
# Hit and runs by state of occurance
query = """SELECT state_number, COUNT(consecutive_number) as hit_and_runs
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run="Yes" 
            GROUP BY state_number
            ORDER By 2 DESC
        """
state_hrs2 = accidents.query_to_pandas_safe(query)
print(state_hrs2.head(10))