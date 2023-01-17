# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head('accident_2015').iloc[:,'hit_and_run']
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
accidents_by_day.head()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
query_1= """SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour, COUNT(consecutive_number) AS accidents_num
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY 1
            ORDER BY 1 ASC            
            """
accidents_by_hour = accidents.query_to_pandas_safe(query_1)
accidents_by_hour
fig = plt.figure(figsize = (10,5))
ax = fig.add_subplot(111)
ymax = max(accidents_by_hour.accidents_num)
xpos = accidents_by_hour.accidents_num.argmax()
ax.annotate('Local Max', xy = (xpos, ymax))
plt.plot(accidents_by_hour.accidents_num)
query_2= """SELECT registration_state_name, COUNT(hit_and_run) AS hit_run_num
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY 1
            ORDER BY 2 DESC           
            """
hits_states = accidents.query_to_pandas_safe(query_2)
hits_states