# import package with helper functions 
import bq_helper
import pandas as pd

# create a helper object for this dataset
open_traffic = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to SELECT the count of hours, ordered by descending frequency
query = """SELECT hour_of_crash, COUNT(hour_of_crash) as hour_freq_2015
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour_of_crash
            ORDER BY hour_of_crash
        """

query2 = """SELECT hour_of_crash, COUNT(hour_of_crash) as hour_freq_2016
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hour_of_crash
            ORDER BY hour_of_crash 
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
most_accidents = open_traffic.query_to_pandas_safe(query)
most_accidents2 = open_traffic.query_to_pandas_safe(query2)
# joining the data in python
most_accidents_total = pd.merge(most_accidents, most_accidents2, on='hour_of_crash')
most_accidents_total['total_count'] = most_accidents_total['hour_freq_2015'] + most_accidents_total['hour_freq_2016']
result = most_accidents_total.sort_values(by=['total_count'], ascending=[False])
result.head()
# query to SELECT the count of accidents by state
query3 = """SELECT registration_state_name, hit_and_run, SUM(vehicle_number) as total_vehicles_2015
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name, hit_and_run
            HAVING hit_and_run = 'Yes'
            ORDER BY registration_state_name
        """

query4 = """SELECT registration_state_name, hit_and_run, SUM(vehicle_number) as total_vehicles_2016
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY registration_state_name, hit_and_run
            HAVING hit_and_run = 'Yes'
            ORDER BY registration_state_name
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
state_accidents = open_traffic.query_to_pandas_safe(query3)
state_accidents2 = open_traffic.query_to_pandas_safe(query4)
# joining the data in python
state_accidents_total = pd.merge(state_accidents, state_accidents2, on='registration_state_name')
state_accidents_total['total_count'] = state_accidents_total['total_vehicles_2015'] + state_accidents_total['total_vehicles_2016']
result2 = state_accidents_total.sort_values(by=['total_count'], ascending=[False])
result2.head()