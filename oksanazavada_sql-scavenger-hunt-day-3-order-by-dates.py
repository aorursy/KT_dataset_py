# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.list_tables()

accidents.head("accident_2015")
accidents.table_schema("accident_2015")

accidents.head("vehicle_2015")
accidents.table_schema("vehicle_2015")

# query to find out the number of accidents which 
# happen on each hour of the days
query_hourofdays = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

# query to find out the number of accidents which 
# happen on each hour of the days
query_state_most_hit_runs = """SELECT COUNT(hit_and_run), 
                  registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name 
            ORDER BY COUNT(hit_and_run) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query_hourofdays)
accidents_by_state = accidents.query_to_pandas_safe(query_state_most_hit_runs)

# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour in 2015\n (Most to least dangerous)")
plt.show()
plt.close('all')
print(accidents_by_hour)

plt.plot(accidents_by_state.f0_)
plt.title("Number of Accidents by Rank of State in 2015\n (Most to least dangerous)")

print(accidents_by_state)