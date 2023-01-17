# import package with helper functions
import bq_helper
import matplotlib.pyplot as plt

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project ="bigquery-public-data", dataset_name = "nhtsa_traffic_fatalities")
# query to find out the number of accidents which happen
# on each day of the week
query = """SELECT COUNT(consecutive_number) as Number, EXTRACT(DAYOFWEEK FROM timestamp_of_crash) as to_day
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY to_day
           ORDER BY Number DESC
        """
accidents_by_day = accidents.query_to_pandas_safe(query)
print(accidents_by_day)
# a plot to show that the data is sorted
plt.plot(accidents_by_day.Number)
plt.title("Number of accidents by rank of day \n (Most to least dangerous)")
# which hour of the day do the most accidents occur
per_hour = """SELECT COUNT(consecutive_number) as Number, EXTRACT(HOUR FROM timestamp_of_crash) as to_hour
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
              GROUP BY to_hour
              ORDER BY Number DESC
           """

accidents_per_hour = accidents.query_to_pandas_safe(per_hour)
print(accidents_per_hour)
# Which state has the most hit and runs?
hits = """SELECT registration_state_name, COUNT(vehicle_number) as Number
          FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
          WHERE hit_and_run = "Yes"
          GROUP BY registration_state_name
          ORDER BY Number DESC
"""
hit_and_run = accidents.query_to_pandas_safe(hits)
print(hit_and_run)
