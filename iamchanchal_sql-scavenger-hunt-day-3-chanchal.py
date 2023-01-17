# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
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
plt.show()
print(accidents_by_day)
## Which hours of the day do the most accidents occur during?

query_day_hour = """SELECT 
                  EXTRACT(Date FROM timestamp_of_crash) Date ,
                  EXTRACT(hour FROM timestamp_of_crash) Hour ,
                  COUNT(1) No_Of_Accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(Date FROM timestamp_of_crash) , EXTRACT(hour FROM timestamp_of_crash)
        """
accidents_day_hour = accidents.query_to_pandas_safe(query_day_hour)
accidents_day_hour.sort_values(['Date','No_Of_Accidents'],ascending=[1, 0])
## Which state has the most hit and runs?

query_hit_and_run = """SELECT 
                  state_number,
                  countif(hit_and_run='Yes') count_of_hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            group by state_number
            order by countif(hit_and_run='Yes') desc
        """
hit_and_run = accidents.query_to_pandas_safe(query_hit_and_run)
hit_and_run

