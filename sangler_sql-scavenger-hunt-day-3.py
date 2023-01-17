# SQL Scavenger Hunt Day 3 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # plotting

import bq_helper
# Create a helper object for our big query datasets
USTF = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", \
                                dataset_name="nhtsa_traffic_fatalities")
USTF

#print all the tables in this dataset
USTF.list_tables()
# print information on all columns in the 'accident_2015' table
USTF.table_schema('accident_2015')
# print information on all columns in the 'accident_2016' table
USTF.table_schema('accident_2016')
USTF.head('accident_2015')
# Question 1 for 2015
query11 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour,
                    COUNT(consecutive_number) AS count
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             GROUP BY hour
             ORDER BY count DESC
             """
q11 = USTF.query_to_pandas_safe(query11, max_gb_scanned=0.1)
q11.head()
sns.set_style("whitegrid")
ax = sns.barplot(x="hour", y="count", data=q11, color="salmon", saturation=.5) \
     .set_title('Distribution of Accidents By Hours in Year 2015')
# Question 1 for 2016
query12 = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour, 
                    COUNT(consecutive_number) AS count
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             GROUP BY hour
             ORDER BY count DESC
             """

q12 = USTF.query_to_pandas_safe(query12, max_gb_scanned=0.1)
q12.head()
sns.set_style("whitegrid")
ax = sns.barplot(x="hour", y="count", data=q12, color="salmon", saturation=.5) \
        .set_title('Distribution of Accidents By Hours in Year 2016')
# Question 2 for 2015
query21 = """SELECT DISTINCT registration_state_name, 
                    COUNT(consecutive_number) AS count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY count DESC
            """
q21 = USTF.query_to_pandas_safe(query21, max_gb_scanned=0.1)
q21.head()
sns.set_style("whitegrid")
sns.set(font_scale = 0.8)
ax = sns.barplot(x="registration_state_name", y="count", data=q21, color="salmon", saturation=.5)
ax.set_title('Distribution of Hit-And-Run By State in Year 2015')

for item in ax.get_xticklabels():
    item.set_rotation(90)
# Question 2 for 2016
query22 = """SELECT DISTINCT registration_state_name, 
                    COUNT(consecutive_number) AS count
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
             WHERE hit_and_run = 'Yes'
             GROUP BY registration_state_name
             ORDER BY count DESC
             """
q22 = USTF.query_to_pandas_safe(query22, max_gb_scanned=0.1)
q22.head()
sns.set_style("whitegrid")
sns.set(font_scale = 0.8)
ax = sns.barplot(x="registration_state_name", y="count", data=q22, color="salmon", saturation=.5)
ax.set_title('Distribution of Hit-And-Run By State in Year 2016')

for item in ax.get_xticklabels():
    item.set_rotation(90)