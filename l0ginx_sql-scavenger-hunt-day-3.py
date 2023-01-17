# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

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


# Use this
print(accidents_by_day)

# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
plt.plot(accidents_by_day.f0_)


query2 = """SELECT
                  EXTRACT(HOUR FROM timestamp_of_crash),
                  COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query2)

plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous 2016)")
plt.plot(accidents_by_hour.f1_)

print(accidents_by_hour)


query3 = """SELECT
                  registration_state_name
                  ,hit_and_run
                  ,COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name, hit_and_run
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_hit_and_run = accidents.query_to_pandas_safe(query3)
accidents_hit_and_run




import seaborn as sns

f, ax = plt.subplots(figsize=(6, 15))
sns.set_style("whitegrid") 

ax = sns.barplot(x="f0_", y="registration_state_name", data=accidents_hit_and_run,
                 palette='coolwarm', dodge=False)


