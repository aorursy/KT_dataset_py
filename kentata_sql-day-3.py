# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                     dataset_name="nhtsa_traffic_fatalities")
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
           ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_day = accidents.query_to_pandas_safe(query)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set();

plt.bar(range(len(accidents_by_day.f1_.values)),  accidents_by_day.f0_.values,
        tick_label=accidents_by_day.f1_.values)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")

accidents_by_day
accidents.list_tables()
accidents.table_schema("accident_2016")
accidents.head("accident_2016")
# Question 1
# Which hours of the day do the most accidents occur during?
query_1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
           GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
           ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query_1)
accidents_by_hour.head()
# sanity check: there are 24 hours in a day. There should be 24 rows in the table
accidents_by_hour.shape
plt.bar(range(len(accidents_by_hour.f1_.values)),  accidents_by_hour.f0_.values,
        tick_label=accidents_by_hour.f1_.values)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
# Question 2
# Which state has the most hit and runs?

accidents.table_schema("vehicle_2016")
accidents.head("vehicle_2016")
# Question 2
# Which state has the most hit and runs?

query_2 = """SELECT registration_state_name, COUNT(consecutive_number)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             GROUP BY registration_state_name 
             ORDER BY COUNT(consecutive_number) DESC
          """
state_of_hit_and_run = accidents.query_to_pandas_safe(query_2)
state_of_hit_and_run.head()
plt.figure(figsize=(20, 20))
plt.barh(range(len(state_of_hit_and_run.f0_.values)), state_of_hit_and_run.f0_.values[::-1], 
         tick_label=state_of_hit_and_run.registration_state_name.values[::-1])
plt.title("The number of hit and run by state in descending order")
