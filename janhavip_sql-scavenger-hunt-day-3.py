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

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
#We now create a query to extract information on the number of accidents occuring during each hour of the day in 2015.
query1 = """SELECT COUNT(consecutive_number) as Count, EXTRACT(HOUR from timestamp_of_crash) as Hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour
            ORDER BY Count DESC
            """

accidents_per_hour = accidents.query_to_pandas_safe(query1)
accidents_per_hour
#We see the distribution of accidents by hour in a descending order.
#Notice how the 'Hour 18' and 'Hour 20' seem to show maximum accident count while 'Hour 4' shows least.
#How about plotting it for confirmation?

import matplotlib.pyplot as plt
plt.plot(accidents_per_hour.Count)
plt.title("Accidents per hour in decreasing order")
plt.xlabel("Hour")
plt.ylabel("Accident Count")

#Moving on to the next question.
#We now look for the state that has maximum number of hit and run cases.
query2 = """SELECT COUNT(vehicle_number) as Count, registration_state_name as State
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY State
            ORDER BY Count DESC
            """

hit_and_run_count_by_state = accidents.query_to_pandas_safe(query2)
hit_and_run_count_by_state
#As we walk through the list, you can check how the distribution of hit and run accidents is.


