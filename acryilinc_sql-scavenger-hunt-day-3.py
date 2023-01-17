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
# Query for number of accidents by hour of the day
no_of_accidents_by_hour_of_day = """ SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS Hour,
                    COUNT(consecutive_number) AS No_of_Accidents                    
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                    GROUP BY Hour
                    ORDER BY COUNT(consecutive_number) DESC
                    """

# Runs query safely
accidents_by_hour = accidents.query_to_pandas_safe(no_of_accidents_by_hour_of_day)

# Returns the number of accidents by hour of the day
print(accidents_by_hour)
# Query for number of vehicles in accidents by hour of the day
no_of_hitrun_by_state = """ SELECT registration_state_name AS State,
                    COUNT(consecutive_number) AS No_of_HitRun_Vehicles
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                    WHERE hit_and_run='Yes'
                    GROUP BY State
                    ORDER BY COUNT(consecutive_number) DESC
                    """

# Runs query safely
hitrun_by_state = accidents.query_to_pandas_safe(no_of_hitrun_by_state)

# Returns the number of hit and runs by state
print(hitrun_by_state)