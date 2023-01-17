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
# Your # import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents.estimate_query_size(query)
accidents_by_hour_2015 = accidents.query_to_pandas_safe(query)
print(accidents_by_hour_2015)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
resorted = accidents_by_hour_2015.sort_values(by=['f1_'])
plt.bar(resorted.f1_,resorted.f0_)
plt.title("Number of Accidents by Rank of Hour in 2015")
plt.xlabel("Hour (0th Hour - 24th Hour)")
plt.ylabel("Number of Accidents")
# query to find out the number of accidents which 
# happen on each day of the week
query2 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour_2016 = accidents.query_to_pandas_safe(query2)
print(accidents_by_hour_2016)
resorted = accidents_by_hour_2016.sort_values(by=['f1_'])
plt.bar(resorted.f1_,resorted.f0_)
plt.title("Number of Accidents by Rank of Hour in 2016")
plt.xlabel("Hour (0th Hour - 24th Hour)")
plt.ylabel("Number of Accidents")
# query to find out the number of accidents which 
# happen on each day of the week
query3 = """SELECT COUNT(consecutive_number), 
                  hour_of_crash
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hour_of_crash
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour_2016b = accidents.query_to_pandas_safe(query3)
print(accidents_by_hour_2016b)
drop_99 = accidents_by_hour_2016b.drop(accidents_by_hour_2016b.index[len(accidents_by_hour_2016b)-1])
resorted = drop_99.sort_values(by=['hour_of_crash'])
plt.bar(resorted.hour_of_crash,resorted.f0_)
plt.title("Number of Accidents by Rank of Hour in 2016")
plt.xlabel("Hour (0th Hour - 24th Hour)")
plt.ylabel("Number of Accidents")
query4 = """SELECT hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY hit_and_run
        """
hitandrun_by_state = accidents.query_to_pandas_safe(query4)
print(hitandrun_by_state)
# query to find out the number of accidents which 
# happen on each day of the week
query5 = """SELECT COUNT(hit_and_run), 
                  registration_state_name 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run='Yes'
            GROUP BY registration_state_name 
            ORDER BY COUNT(hit_and_run) DESC
        """
accidents.estimate_query_size(query5)
hitandrun_by_state_2015 = accidents.query_to_pandas_safe(query5)
print(hitandrun_by_state_2015)
# query to find out the number of accidents which 
# happen on each day of the week
query6 = """SELECT COUNT(hit_and_run), 
                  registration_state_name 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run='Yes'
            GROUP BY registration_state_name 
            ORDER BY COUNT(hit_and_run) DESC
        """
hitandrun_by_state_2016 = accidents.query_to_pandas_safe(query6)
print(hitandrun_by_state_2016)
