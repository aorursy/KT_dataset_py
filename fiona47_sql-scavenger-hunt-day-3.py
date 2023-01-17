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
# Your code goes here :)
# import package with helper functions 
import bq_helper
# library for plotting
import matplotlib.pyplot as plt

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                  GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                  ORDER BY COUNT(consecutive_number) DESC
         """

accidents_by_hour = accidents.query_to_pandas_safe(query)

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)

plt.xlabel('Hour of the day', weight='bold', size='large')
plt.ylabel('Number of Accidents', weight='bold', size='large')

plt.xticks(range(len(accidents_by_hour.f0_)), accidents_by_hour.f1_, rotation=90, horizontalalignment='right')
plt.yticks()

plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
plt.show()
print(accidents_by_hour)
query2 ='''Select count(hit_and_run),registration_state_name 
            from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            group by registration_state_name
            order by count(hit_and_run) desc
'''

hit_and_run_by_state=accidents.query_to_pandas_safe(query2)

position = []
for pos in range(0, 51):
    position.append(pos*2)

plt.figure(figsize=(20,7))
plt.plot(hit_and_run_by_state.f0_)
plt.title("Number of Hit and run by Rank of State \n (Most to least Hit and run)")

plt.xlabel('Hour of the day', weight='bold', size='large')
plt.ylabel('Number of Accidents', weight='bold', size='large')

plt.xticks(range(len(hit_and_run_by_state.f0_)), hit_and_run_by_state.registration_state_name, rotation=90, horizontalalignment='right')
plt.yticks()

plt.show()
print (hit_and_run_by_state)