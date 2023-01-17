# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.head("accident_2015")
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
# Which hours of the day do the most accidents occur?
query_by_hour = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query_by_hour)
#rename  columns
accidents_by_hour.columns = ['Accidents happened','Hour of day']
print(accidents_by_hour)

accidents_by_hour.plot.bar(x='Hour of day',y='Accidents happened')
#Which state the most hit and run accidents happened
query_by_state = """SELECT COUNT(hit_and_run), 
                  registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """

hit_and_run_accidents_by_state = accidents.query_to_pandas_safe(query_by_state)
hit_and_run_accidents_by_state.columns =['Hit and run accidents', 'State name']
hit_and_run_accidents_by_state.head()
hit_and_run_accidents_by_state.plot.bar(x='State name', y ='Hit and run accidents')