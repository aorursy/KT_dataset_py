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
plt.xticks(accidents_by_day.index,accidents_by_day.f1_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# form query for question 1
query2 = """SELECT COUNT(consecutive_number) AS accNum, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS hourOfDay
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hourOfDay
            ORDER BY accNum DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query2)

accidents_by_hour.head()
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot((accidents_by_hour.accNum/accidents_by_hour.accNum.sum())*100)
plt.xticks(accidents_by_hour.index,accidents_by_hour.hourOfDay)
plt.title("Accident Rates by Rank of Hour in the US in 2015 \n (Most to least dangerous)")
plt.ylabel("Accident Rates %")
plt.xlabel("Hour of day (0-23)")
# make a plot to show that our data is, actually, sorted:
plt.bar(accidents_by_hour.hourOfDay,(accidents_by_hour.accNum/accidents_by_hour.accNum.sum())*100)
plt.title("Accident Rates by Hour of the Day in the US in 2015")
plt.ylabel("Accident Rates %")
plt.xlabel("Hour of day (0-23)")
# form query for question 2
query3 = """SELECT registration_state_name, COUNT(consecutive_number) AS hitNrunNum
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY hitNrunNum DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_and_runs_per_state = accidents.query_to_pandas_safe(query3)

hit_and_runs_per_state.head()
# make a plot to show that our data is, actually, sorted:
plt.bar(hit_and_runs_per_state.registration_state_name,(hit_and_runs_per_state.hitNrunNum/hit_and_runs_per_state.hitNrunNum.sum())*100)
plt.title("Hit-And-Run Rates by State in the US in 2015")
plt.xticks(rotation=90)
plt.ylabel("Hit-And-Run Rates %")
plt.xlabel("US States")