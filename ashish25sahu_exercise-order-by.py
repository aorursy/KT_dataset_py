# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# accidents.list_tables()
accidents.head("accident_2016")
accidents.table_schema("accident_2015")
query = """SELECT COUNT(consecutive_number), EXTRACT(Hour FROM timestamp_of_crash) FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
            group by EXTRACT(Hour FROM timestamp_of_crash) ORDER BY COUNT(consecutive_number) DESC"""

accidents_by_hour = accidents.query_to_pandas_safe(query)

# print(accidents_by_hour)

import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.bar(accidents_by_hour.f1_, accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")

query = """SELECT COUNT(consecutive_number), state_name  FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` 
            group by state_name ORDER BY COUNT(consecutive_number) DESC"""

accidents_by_state = accidents.query_to_pandas_safe(query)

accidents_by_state.head()
