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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.list_tables()
accidents.head('accident_2016')
# query to find out the number of accidents which 
# happen on each hour
accidents_by_hour = accidents.query_to_pandas_safe(
            """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour,
            COUNT (consecutive_number) AS num_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY HOUR
            ORDER BY num_accidents DESC
        """)

accidents_by_hour
import matplotlib.pyplot as plt
import seaborn as sns

# make a plot to show that our data is, actually, sorted:
plt.figure(figsize=(12,8))
sns.barplot(x = 'hour', y = 'num_accidents', data = accidents_by_hour, color = 'g')
plt.title('Number of Accidents by Hour in 2016')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
print(accidents_by_hour)
accidents.head('vehicle_2016')
# run to view hit and run column
accidents.query_to_pandas_safe("""
SELECT DISTINCT hit_and_run
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
""")
# query to find out the number of accidents which 
# happen on each hour
hit_and_run_accidents = accidents.query_to_pandas_safe(
            """SELECT registration_state_name AS state, 
            COUNT (consecutive_number) AS hit_and_run_accidents
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = 'Yes'
            GROUP BY state
            ORDER BY hit_and_run_accidents DESC
        """)
hit_and_run_accidents
# make a plot to show that our data is, actually, sorted:
plt.figure(figsize=(8,15))
sns.barplot(x = 'hit_and_run_accidents', y = 'state', data = hit_and_run_accidents, 
            orient = 'h', color = 'g')
plt.title('Number of Hit and Runs by State in 2016')
plt.xlabel('Number of Hit and Runs')
plt.ylabel('State')