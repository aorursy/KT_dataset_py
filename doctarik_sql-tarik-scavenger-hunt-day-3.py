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
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                     dataset_name="nhtsa_traffic_fatalities")
# query to find out the number of accidents which happen on each hour of the day
query = """SELECT COUNT(consecutive_number) as Nbr_Acc, 
            EXTRACT(HOUR FROM timestamp_of_crash) as Hours
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY Hours
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
#print(accidents_by_hour)
accidents_by_hour.head()
# save our dataframe as a .csv 
#accidents_by_hour.to_csv("accidents_by_hour.csv")
import matplotlib.pyplot as plt
plt.plot(accidents_by_hour.Hours,accidents_by_hour.Nbr_Acc)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
plt.xlabel('Hours (h)')
plt.ylabel('Nbr_Accidents')
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                     dataset_name="nhtsa_traffic_fatalities")
query = """SELECT registration_state_name as State, 
            COUNT(consecutive_number) as nbr_vehi_hit_and_run 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
            GROUP BY State
            ORDER BY COUNT(consecutive_number) DESC
        """
States_has_the_most_hit_and_run = accidents.query_to_pandas_safe(query)
States_has_the_most_hit_and_run.head()
#print(States_has_the_most_hit_and_run)
#result.f0_.sum()
# save our dataframe as a .csv 
#States_has_the_most_hit_and_run.to_csv("States_has_the_most_hit_&_run.csv")