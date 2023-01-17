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
accidents_by_day = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
q2 = """SELECT COUNT(consecutive_number) as ACCIDENTS, 
                  EXTRACT(HOUR FROM timestamp_of_crash) as HOUR
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY HOUR
            ORDER BY HOUR ASC
        """
accidents_by_hour = accidents.query_to_pandas_safe(q2)
print(accidents_by_hour)
plt.plot(accidents_by_hour.ACCIDENTS)
plt.ylabel('Accidents')
plt.xlabel('Hours of the day')
plt.title("Number of Accidents by Hour in 2015")
accidents.head("vehicle_2015")


q3= """SELECT count(hit_and_run) as HAR, 
            registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes" AND registration_state_name != "Unknown"
            GROUP BY registration_state_name
            ORDER BY HAR DESC
        """
hit_and_run_by_state = accidents.query_to_pandas_safe(q3)
hit_and_run_by_state.head(15)
plt.figure(figsize=(14,6))
plt.bar(hit_and_run_by_state.registration_state_name,hit_and_run_by_state.HAR)
plt.ylabel('Hit and Runs')
plt.xlabel('States')
plt.xticks(hit_and_run_by_state.registration_state_name, hit_and_run_by_state.registration_state_name, rotation='vertical')
plt.title("Number of Hit and Runs by State in 2015")
