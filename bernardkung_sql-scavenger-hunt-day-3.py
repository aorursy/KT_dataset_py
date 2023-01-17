# import package with helper functions 
import bq_helper as bqh

# create a helper object for this dataset
accidents = bqh.BigQueryHelper(active_project="bigquery-public-data",
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
accidents.head('accident_2015')
query1 = """
SELECT COUNT(consecutive_number) AS num_accidents, EXTRACT(HOUR FROM timestamp_of_crash) AS hour
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY hour
ORDER BY hour
"""
accidents.estimate_query_size(query1)
accidents_by_hour = accidents.query_to_pandas_safe(query1)
print(accidents_by_hour)
plt.plot(accidents_by_hour.num_accidents)
plt.title("Accidents by Hour (2015)")
plt.xlabel("Hour of Day")
plt.ylabel("Accidents per Hour")
accidents.head('vehicle_2016')
accidents.table_schema('vehicle_2016')
query2a = """
SELECT DISTINCT(hit_and_run)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
"""
accidents.query_to_pandas_safe(query2a)
query2b = """
SELECT state_number, COUNT(hit_and_run) AS count_hit_and_run
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
WHERE hit_and_run = "Yes"
GROUP BY state_number
ORDER BY count_hit_and_run DESC
"""
hit_and_run_by_state = accidents.query_to_pandas_safe(query2b)
print(hit_and_run_by_state)