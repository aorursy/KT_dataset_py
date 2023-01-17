# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# accidents.list_tables()
# accidents.table_schema("accident_2015")
accidents.head("accident_2015")

query = """SELECT COUNT(consecutive_number) as amount, EXTRACT(DAYOFWEEK FROM timestamp_of_crash) as day
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY day
ORDER BY amount DESC
"""
accidents_by_day = accidents.query_to_pandas_safe(query, max_gb_scanned=0.1)
print(accidents_by_day)
# library for plotting
import matplotlib.pyplot as plt
# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.amount)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
query = """SELECT COUNT(consecutive_number) as amount, EXTRACT(DAYOFWEEK FROM timestamp_of_crash) as day
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY day
ORDER BY amount DESC
"""
accidents_by_day = accidents.query_to_pandas_safe(query, max_gb_scanned=0.1)
accidents.table_schema("vehicle_2015")
accidents.head("vehicle_2015")
query = """SELECT COUNT(consecutive_number) as amount, registration_state_name as state
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
WHERE hit_and_run = "Yes"
GROUP BY state
ORDER BY amount DESC
"""
#accidents.estimate_query_size(query)
state_hr = accidents.query_to_pandas_safe(query, max_gb_scanned=0.1)
print(state_hr)