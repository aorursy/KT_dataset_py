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
# query to find out the number of accidents which 
# happen on each hour of the day
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as Hour_of_day,
            COUNT(consecutive_number) as accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE EXTRACT(YEAR FROM timestamp_of_crash) = 2015
            GROUP BY Hour_of_day
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour_2015 = accidents.query_to_pandas_safe(query)
accidents_by_hour_2015

# query to find out the number of accidents which 
# happen by state
query = """SELECT registration_state_name,
            COUNT(consecutive_number) as hit_run_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE EXTRACT(YEAR FROM timestamp_of_crash) = 2015
            AND hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hr_accidents_by_vehicle_state_2015 = accidents.query_to_pandas_safe(query)
hr_accidents_by_vehicle_state_2015
