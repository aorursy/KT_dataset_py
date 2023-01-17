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
accidents_by_day
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_,accidents_by_day.f1_, '.')
plt.ylabel( "1=Sunday thru 7=Saturday")
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
query = """SELECT EXTRACT( HOUR FROM `timestamp_of_crash`), COUNT(*) as HOURLY_EVENTS
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
           GROUP BY EXTRACT( HOUR FROM `timestamp_of_crash`)
           ORDER BY HOURLY_EVENTS DESC"""

accidents.estimate_query_size(query)
hourly_accidents = accidents.query_to_pandas_safe(query)
hourly_accidents.head()
#query = """SELECT registration_state_name, hit_and_run, COUNT(*) AS state_hit_and_runs
#           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
#           GROUP BY registration_state_name, hit_and_run
#           HAVING hit_and_run = '1'
#           ORDER BY state_hit_and_runs DESC"""
query = """SELECT count(*)
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
           WHERE hit_and_run = 'Yes'"""
accidents.estimate_query_size(query)
hit_and_runs_by_state = accidents.query_to_pandas_safe(query)
hit_and_runs_by_state.head()
