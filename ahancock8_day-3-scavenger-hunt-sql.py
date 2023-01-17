# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.list_tables()

# Question 1
accidents.head('accident_2015')

acci="""SELECT COUNT(1),
        EXTRACT (HOUR FROM timestamp_of_crash)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
        ORDER BY COUNT(1) DESC"""

accidents_by_day = accidents.query_to_pandas_safe(acci)

accidents_by_day.head()

# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Hour of Day \n (Most to least dangerous)")

print(accidents_by_day)

# End Question 1

accidents.list_tables()

# Question 2

accidents.head('vehicle_2015')

hr = """select count(1), registration_state_name, count(hit_and_run) 
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        GROUP BY registration_state_name
        ORDER BY count(hit_and_run) DESC"""

hit_and_run = accidents.query_to_pandas_safe(hr)

plt.plot(hit_and_run.f0_)
plt.title("Hit and Runs by State \n (Most to least dangerous)")

print(hit_and_run)

