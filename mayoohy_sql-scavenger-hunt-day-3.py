# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents.head("accident_2015")
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
                                    dataset_name ="nhtsa_traffic_fatalities")
accidents.head("accident_2016")
query1 = """ SELECT COUNT(consecutive_number),
                    EXTRACT(HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query1)

# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")

print(accidents_by_hour)
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                    dataset_name='nhtsa_traffic_fatalities')
accidents.head("vehicle_2016")
query2 = """ SELECT COUNT(consecutive_number), registration_state
              FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
              GROUP BY hit_and_run, registration_state
              HAVING hit_and_run = 'Yes'
              ORDER BY COUNT(consecutive_number) DESC
         """
hit_and_run_by_state = accidents.query_to_pandas_safe(query2)

#library for plotting
import matplotlib.pyplot as plt

#make a plot to show that our data is actually sorted
plt.plot(hit_and_run_by_state.f0_)

print(hit_and_run_by_state)