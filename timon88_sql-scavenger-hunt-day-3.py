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
query_hour_2015 = """SELECT COUNT(consecutive_number) as Number_of_accidents, 
                     EXTRACT(HOUR FROM timestamp_of_crash) as Hour
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                     GROUP BY Hour
                     ORDER BY Number_of_accidents DESC
                  """
accidents_by_hour_2015 = accidents.query_to_pandas_safe(query_hour_2015)
print(accidents_by_hour_2015)
query_hour_2016 = """SELECT COUNT(consecutive_number) as Number_of_accidents, 
                     EXTRACT(HOUR FROM timestamp_of_crash) as Hour
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                     GROUP BY Hour
                     ORDER BY Number_of_accidents DESC
                  """
accidents_by_hour_2016 = accidents.query_to_pandas_safe(query_hour_2016)
print(accidents_by_hour_2016)
plt.bar(accidents_by_hour_2015.Hour, accidents_by_hour_2015.Number_of_accidents)
plt.title("Number of Accidents by Hour in 2015 \n (Most to least dangerous)")
plt.bar(accidents_by_hour_2016.Hour, accidents_by_hour_2016.Number_of_accidents)
plt.title("Number of Accidents by Hour in 2016 \n (Most to least dangerous)")
query_hit_and_run_2015 = """SELECT registration_state_name, 
                               COUNT(hit_and_run) as Hit_and_run
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                        WHERE hit_and_run = "Yes"
                        GROUP BY registration_state_name
                        ORDER BY Hit_and_run DESC
                     """
Hit_and_run_2015 = accidents.query_to_pandas_safe(query_hit_and_run_2015)
print(Hit_and_run_2015)    
plt.figure(figsize=(6,10))
plt.barh(Hit_and_run_2015.registration_state_name, Hit_and_run_2015.Hit_and_run)
plt.title("Number of vehicles registered in each state in 2015")
query_hit_and_run_2016 = """SELECT registration_state_name, 
                               COUNT(hit_and_run) as Hit_and_run
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                        WHERE hit_and_run = "Yes"
                        GROUP BY registration_state_name
                        ORDER BY Hit_and_run DESC
                     """
Hit_and_run_2016 = accidents.query_to_pandas_safe(query_hit_and_run_2016)
print(Hit_and_run_2016)  
plt.figure(figsize=(6,10))
plt.barh(Hit_and_run_2016.registration_state_name, Hit_and_run_2016.Hit_and_run)
plt.title("Number of vehicles registered in each state in 2016")