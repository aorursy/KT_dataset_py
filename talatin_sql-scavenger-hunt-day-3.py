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
#query to find out the numbers of accident happened
#in which hour in the day
query = """ SELECT COUNT(consecutive_number),
            EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT (consecutive_number) DESC
            """
  
#safe query 
Accidents_by_hour = accidents.query_to_pandas_safe(query)
#print result 
Accidents_by_hour


#query to find out which satate has most hit and run accidents
query1 = """SELECT registration_state_name, 
                COUNT(hit_and_run) as Count
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                where hit_and_run = 'Yes'
                GROUP BY registration_state_name
                ORDER BY COUNT(hit_and_run) DESC
                """
#safe query
hit_and_run_by_state = accidents.query_to_pandas_safe(query1)
#print result
hit_and_run_by_state