# importing packages with helper fuctions
import bq_helper #helper for BigQuary

# library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which 
# happen on which hours of the day 
query1 = """SELECT COUNT(consecutive_number),
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash) 
            ORDER BY COUNT(consecutive_number) DESC 
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query1)
accidents_by_hour

plt.rcParams["figure.figsize"] = (12, 8)
ax = sns.barplot(x="f1_", y="f0_", data=accidents_by_hour,palette='coolwarm')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.title("Number of Accidents by Rank of Hour")
plt.show()
query3 = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
            """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_run_by_state = accidents.query_to_pandas_safe(query3)
hit_run_by_state
f, ax = plt.subplots(figsize= (12, 8))
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") 
ax = sns.barplot(x="f0_", y="registration_state_name", data=hit_run_by_state,palette='coolwarm',dodge=False)
plt.tight_layout()
plt.title("Number of hit and run of each state")
plt.show()