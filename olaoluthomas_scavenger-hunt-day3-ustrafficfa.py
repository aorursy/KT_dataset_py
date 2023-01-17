# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="nhtsa_traffic_fatalities")

accidents.head("accident_2015")
# number of accidents that occur each day of week
query_1 = """SELECT COUNT(consecutive_number),
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
           ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_day = accidents.query_to_pandas_safe(query_1)
import matplotlib.pyplot as plt

plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
query_2 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour_of_day,
                    COUNT(consecutive_number) AS count
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             GROUP BY hour_of_day
             ORDER BY count DESC
            """
accidents_by_hour = accidents.query_to_pandas_safe(query_2)
accidents_by_hour
accidents.head("vehicle_2015")
query_3 = """SELECT registration_state_name AS state, COUNT(hit_and_run) AS count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY state
            ORDER by count DESC
            """
hit_and_run = accidents.query_to_pandas_safe(query_3)
hit_and_run
query_3 = """SELECT registration_state_name AS state, COUNT(hit_and_run) AS count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY state
            ORDER by count DESC
            """
hit_and_run_2 = accidents.query_to_pandas_safe(query_3)
hit_and_run_2
