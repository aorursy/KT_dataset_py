# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number) accidents_count, 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) day_of_week
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY day_of_week
            ORDER BY accidents_count DESC
        """
# check query size
accidents.estimate_query_size(query)
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
# print the first few rows of the table
accidents_by_day.head()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.accidents_count)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# query to find out the number of accidents which 
# happen on each hour of the day in 2015, sorted by same
query2 = """SELECT COUNT(consecutive_number) accidents_count, 
                  EXTRACT(HOUR FROM timestamp_of_crash) hour_of_week
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour_of_week
            ORDER BY accidents_count DESC
        """
# check query size
accidents.estimate_query_size(query2)
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query2)
# print the first few rows of the table
accidents_by_hour.head()
# query to find out the number of registered vehicles in each state that were 
# were involved in hit-and-run accidents, sorted by number of hit-and-runs.
query3 = """SELECT COUNT(consecutive_number) accidents_count, 
                  registration_state_name registration_state
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state, hit_and_run
            HAVING hit_and_run = "Yes"
            ORDER BY accidents_count DESC
        """
# check query size
accidents.estimate_query_size(query3)
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
number_hit_runs = accidents.query_to_pandas_safe(query3)
# print the first few rows of the table
number_hit_runs.head()
number_hit_runs.to_csv("hit_and_run_by_state.csv")