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
#accidents.head('accident_2015')
list(accidents.head('accident_2015'))



query3 = """
        select EXTRACT(HOUR FROM timestamp_of_crash) as Hour_of_Accident,
        COUNT(consecutive_number) as Accidents_Count
        from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        group by Hour_of_Accident
        order by Accidents_Count desc
        """
crashes = accidents.query_to_pandas_safe(query3)

crashes.head(20)
# hit and runs by state
state_query = """SELECT registration_state_name as state,
            COUNT(hit_and_run) as hit_and_run_count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY State
            ORDER BY hit_and_run_count DESC
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(state_query)
print(hit_and_run_by_state)