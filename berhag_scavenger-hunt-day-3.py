import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

import bq_helper
traffic_fatality_records = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                         dataset_name = "nhtsa_traffic_fatalities")
traffic_fatality_records.list_tables()
traffic_fatality_records.head('accident_2015', num_rows = 5)
query = """ SELECT state_name, number_of_fatalities,number_of_drunk_drivers, timestamp_of_crash
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            ORDER BY number_of_fatalities DESC
            """
df = traffic_fatality_records.query_to_pandas_safe(query)
df.head()
query = """ SELECT state_name, number_of_fatalities,number_of_drunk_drivers, timestamp_of_crash
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            ORDER BY number_of_drunk_drivers DESC
            """
df = traffic_fatality_records.query_to_pandas_safe(query)
df.head()
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
df = traffic_fatality_records.query_to_pandas_safe(query)
df.head()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(df.f0_)
plt.title("Number of Accidents by decreasing order)")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
df = traffic_fatality_records.query_to_pandas_safe(query)
df.head()
plt.plot(df.f0_)
plt.title("Number of hourly Accidents by decreasing order")
query = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name, hit_and_run
            HAVING hit_and_run = 'Yes'
            ORDER BY COUNT(hit_and_run) DESC
        """
df = traffic_fatality_records.query_to_pandas_safe(query)
df.head(10)
plt.plot(df.f0_)
plt.title("Number of hit and run accidents per state in 2015")

