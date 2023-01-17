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
query = """(SELECT EXTRACT(day FROM timestamp_of_crash) AS day ,EXTRACT(HOUR FROM timestamp_of_crash) AS hour, COUNT(consecutive_number) AS value
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY day,hour
            ORDER BY COUNT(consecutive_number) DESC)
        """

query2 = """(SELECT day, MAX(value) AS peak FROM """ +query+""" GROUP BY day)"""

query3 = """SELECT x.day,x.hour FROM """+query+""" x INNER JOIN """+query2+""" y ON x.value=y.peak AND x.day=y.day ORDER BY day DESC"""

query4 = """(SELECT day,hour,MAX(value) AS peak FROM """ +query+""" GROUP BY day,hour ORDER BY day,peak DESC)"""
accidents_by_hour_total = accidents.query_to_pandas_safe(query4);
accidents_by_hour_total
accidents_by_hour = accidents.query_to_pandas_safe(query3);
accidents_by_hour.sort_values(by='day',inplace=True)
accidents_by_hour.reset_index(drop=True)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:

plt.scatter(accidents_by_hour['day'],accidents_by_hour['hour'])
plt.plot(accidents_by_hour['day'],accidents_by_hour['hour'])
plt.xlabel("Days of month")
plt.ylabel("Hours of accident")
# Your code goes here :)
query5 = """SELECT registration_state_name, COUNT(consecutive_number) AS number_of_cases ,SUM(vehicle_number) as number_of_vehicles_involved
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes' AND registration_state_name != 'Unknown'
            GROUP BY registration_state_name 
            ORDER BY number_of_vehicles_involved DESC
            
        """
accidents_by_state = accidents.query_to_pandas_safe(query5);
accidents_by_state