import bq_helper

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
accidents_by_day.head()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
per_hour = """SELECT COUNT(consecutive_number) as accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)  
            ORDER BY COUNT(consecutive_number) DESC 
        """

accidents_per_hour = accidents.query_to_pandas_safe(per_hour)

accidents_per_hour.head()
plt.plot(accidents_per_hour.f0_)
plt.title('Hourly accidents in 2015')
har = """SELECT COUNT(hit_and_run), registration_state_name
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
           GROUP BY registration_state_name
           ORDER BY COUNT(hit_and_run) DESC
       """
states_har = accidents.query_to_pandas_safe(har)
states_har.head()
