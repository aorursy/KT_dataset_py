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
# print all the tables in this dataset
accidents.list_tables()
# print the first couple rows of the "accident_2016" table
accidents.head("accident_2016")
# query to find out the number of accidents which 
# happen on each hour of the day
query_hour = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# check how big this query will be
accidents.estimate_query_size(query_hour)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
hour_accidents = accidents.query_to_pandas_safe(query_hour)
hour_accidents
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(hour_accidents.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
accidents.table_schema("vehicle_2016")
accidents.head("vehicle_2016", selected_columns="hit_and_run, registration_state_name", num_rows=10)
# query to find out the number of hit and run accidents
query_hit_and_run = """SELECT COUNT(consecutive_number), 
                  registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """
# check how big this query will be
accidents.estimate_query_size(query_hit_and_run)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
hit_and_run_accidents = accidents.query_to_pandas_safe(query_hit_and_run)
hit_and_run_accidents