

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                    dataset_name="nhtsa_traffic_fatalities")
query =  """SELECT COUNT(consecutive_number),
                   EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC

         """
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day.head()
import matplotlib.pyplot as plt

plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
query_hr = """SELECT COUNT(consecutive_number),
                     EXTRACT(HOUR FROM timestamp_of_crash)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
           ORDER BY COUNT(consecutive_number) DESC

           """
accidents_by_hour = accidents.query_to_pandas_safe(query_hr)
accidents_by_hour.head()
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour")
accidents.list_tables()
accidents.head("vehicle_2015")
accidents.table_schema("vehicle_2015")
accidents.head("accident_2016")
query_hit = """SELECT registration_state_name AS `State`,
                    COUNT(consecutive_number) AS `HitAndRuns`
                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                 WHERE hit_and_run LIKE "Yes" AND registration_state_name not LIKE "Unknown"
                 GROUP BY registration_state_name
                 ORDER BY COUNT(consecutive_number) DESC

            """
hit_and_run = accidents.query_to_pandas_safe(query_hit)
hit_and_run.head()