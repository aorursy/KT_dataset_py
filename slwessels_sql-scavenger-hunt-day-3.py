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
#accidents.estimate_query_size(query)
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Your code goes here :)
#Which hours of the day do the most accidents occur during?

hourQuery = """SELECT COUNT(consecutive_number) AS Number, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS Hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour
            ORDER BY Number DESC
        """

accidentsByHour = accidents.query_to_pandas_safe(hourQuery)
sortedByHour = accidentsByHour.sort_values("Hour")
sortedByHour.plot(x=sortedByHour.Hour)
#plt.plot(sortedByHour.Number)
#plt.title("Number of Accidents by Hour of the Day\n (Most to least dangerous)")
#accidents.list_tables()
accidents.table_schema('accident_2015')
accidents.head('accident_2015')
accidents.table_schema('accident_2015')
df = accidents.head('vehicle_2015')
df.hit_and_run
df.registration_state_name
#registration_state_name and hit_and_run
#Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state that were involved in 
#hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 
#or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.


hitAndRunQuery = """SELECT registration_state_name, COUNT(registration_state_name) AS stateCount 
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY stateCount DESC"""
hitAndRun = accidents.query_to_pandas_safe(hitAndRunQuery)
hitAndRun