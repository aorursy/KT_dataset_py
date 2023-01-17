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
import bq_helper

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                     dataset_name="nhtsa_traffic_fatalities")

query = """ SELECT COUNT(consecutive_number), 
                   EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

df = accidents.query_to_pandas_safe(query)
import seaborn as sns
import numpy as np

pal = sns.color_palette('coolwarm', len(df))
ax = sns.barplot(x="f1_", y="f0_", data=df, palette=np.array(pal[::-1])[df.f1_.argsort()])
query = """ SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
df2 = accidents.query_to_pandas_safe(query)
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(6, 15))
pal = sns.color_palette('coolwarm', len(df2))
ax = sns.barplot(x="f0_", y="registration_state_name", data=df2, 
                 palette=np.array(pal[::-1])[df2.f0_.argsort()[::-1]],dodge=False)