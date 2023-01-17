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
## Which hours of the day do the most accidents occur during?
## Import bq_helper
import bq_helper
## Create bq_helper object
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="nhtsa_traffic_fatalities")
## Query
query = """SELECT COUNT(consecutive_number) AS n_accidents, EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_crash
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY hour_of_crash
ORDER BY n_accidents DESC;"""

## Query to pandas safe
accidents_by_hour = accidents.query_to_pandas_safe(query)
## Print accidents_by_hour
accidents_by_hour
## The most of accidents occur between 6 PM and 9 PM
## Which State has the most hit and runs?
vehicles = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="nhtsa_traffic_fatalities")
## Query
query = """SELECT COUNT(DISTINCT vehicle_identification_number_vin) AS n_vehicles,
registration_state_name AS state
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
WHERE hit_and_run = "Yes" AND registration_state_name != "Unknown"
GROUP BY state
ORDER BY n_vehicles DESC;
"""
## Query to pandas
n_vehicles = vehicles.query_to_pandas_safe(query)
## Print n_vehicles
n_vehicles
## California has the highest number of unique vehicles involved in hit and runs!