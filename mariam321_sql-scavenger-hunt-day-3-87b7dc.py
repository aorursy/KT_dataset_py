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
#creat helper object
import bq_helper as bq
Traffic_set_helper=bq.BigQueryHelper(active_project="bigquery-public-data",
                                     dataset_name="nhtsa_traffic_fatalities")
#querey to find which hour of the daydo most accidents accur during
query=""" SELECT EXTRACT(HOUR FROM timestamp_of_crash) as Hour, count(consecutive_number)
          from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
          group by Hour
          order by count(consecutive_number) desc """
#execute the query

traffic_report=Traffic_set_helper.query_to_pandas_safe(query)
#print(traffic_report)
####################
#which state has the most hit and run
query2=""" SELECT COUNT(hit_and_run), registration_state_name
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
          WHERE hit_and_run="Yes"
          GROUP BY registration_state_name
          order by COUNT(hit_and_run) desc
           """
state_hit_and_run=Traffic_set_helper.query_to_pandas_safe(query2)
print(state_hit_and_run)



