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
# hour of day
query3A =     """ 
                                SELECT  DATE(timestamp_of_crash),
                                        EXTRACT(HOUR FROM timestamp_of_crash) AS `Hour`,
                                        COUNT(consecutive_number) AS `Accidents`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                GROUP BY  timestamp_of_crash
                                ORDER BY  COUNT(consecutive_number) DESC                                        
                               """
result3A = accidents.query_to_pandas_safe(query3A)
result3A.head(10)



# hit and run states, top
query3B =     """ 
                                SELECT registration_state_name AS `State`,
                                                COUNT(consecutive_number) AS `HitAndRuns`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                WHERE hit_and_run LIKE "Yes" AND registration_state_name NOT LIKE "Unknown"
                                GROUP BY  registration_state_name
                                ORDER BY  COUNT(consecutive_number) DESC                                        
                               """
result3B = accidents.query_to_pandas_safe(query3B)
result3B.head(10)
#read data