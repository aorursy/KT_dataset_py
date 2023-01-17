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
# Daniela's code goes here :)
#Print list of tables
accidents.list_tables()
#Get name of columns for table "accident_2016"
accidents.head("accident_2016")


#1-Which hours of the day do the most accidents occur during?
query_1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
#Safe method
accidents_by_hour = accidents.query_to_pandas_safe(query_1)

print(accidents_by_hour)
#Get name of columns for table "vehicle_2016"
accidents.head("vehicle_2016")
#I want to check which values can be found in 'hit_and_run' column in order to build the WHERE 
#clause in the next section
query_aux = """SELECT hit_and_run,COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY hit_and_run
        """
#Safe method
hit_and_run_categories = accidents.query_to_pandas_safe(query_aux)

print(hit_and_run_categories)
#2-Which state has the most hit and runs?
query_2 = """SELECT registration_state_name,
                  hit_and_run,
                  COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run='Yes'
            GROUP BY registration_state_name,hit_and_run
            ORDER BY COUNT(hit_and_run) DESC
        """
#Safe method
hits_runs_by_state = accidents.query_to_pandas_safe(query_2)

print(hits_runs_by_state)
