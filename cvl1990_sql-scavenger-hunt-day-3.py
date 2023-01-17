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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2015")

query = """SELECT COUNT(consecutive_number) AS ACCIDENTS, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS HOUR
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY HOUR
            ORDER BY ACCIDENTS DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour

# library for plotting
#import matplotlib.pyplot as plt
X=accidents_by_hour.ACCIDENTS
Y=accidents_by_hour.HOUR
# make a plot to show that our data is, actually, sorted:
plt.plot(X,Y)
plt.title("Number of Accidents by Hour \n (Least to most dangerous hour)")

# Which state has the most hit and runs?

accidents.head("vehicle_2015")

query2 = """SELECT registration_state_name as STATE,
            COUNT(hit_and_run) AS HIT 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY STATE
            ORDER BY HIT DESC
        """
STATES = accidents.query_to_pandas_safe(query2)
STATES