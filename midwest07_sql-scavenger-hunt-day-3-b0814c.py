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
accidents.list_tables()
accidents.table_schema("accident_2015")
accidents.head("accident_2015",10)
# Your code goes here :)
query_2=""" 
            select COUNT(consecutive_number), extract(HOUR from timestamp_of_crash) 
            from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            group by extract(hour from timestamp_of_crash)
            order by 1 desc
        """
accidents.estimate_query_size(query_2) 
accidents_count=accidents.query_to_pandas_safe(query_2)
print(accidents_count)
accidents_count.to_csv("accidents_count")
accidents.table_schema("vehicle_2015")
accidents.head("vehicle_2015",1 )
query_3="""
select registration_state_name,hit_and_run ,count(number_of_motor_vehicles_in_transport_mvit)
from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
group by  registration_state_name,hit_and_run 
having hit_and_run = "Yes"
order by count(number_of_motor_vehicles_in_transport_mvit) desc
"""
accidents.estimate_query_size(query_3)
vehicles=accidents.query_to_pandas_safe(query_3)
print(vehicles)
vehicles.to_csv("vehicles")