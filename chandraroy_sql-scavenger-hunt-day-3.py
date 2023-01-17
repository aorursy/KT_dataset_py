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
# Your code goes here :)
import bq_helper 
# Create the bigquery object.
accident = bq_helper.BigQueryHelper( active_project = "bigquery-public-data",
                                    dataset_name = "nhtsa_traffic_fatalities"
                                    )
# Check the tables in the dataset"nhtsa_traffic_fatalities"
accident.list_tables()
# Check columns of target table i.e. accident_2015
accident.table_schema("accident_2015")
# Check first 5 entries usinf head()
accident.head("accident_2015")
query = """ SELECT COUNT(consecutive_number), EXTRACT( HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT( HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(timestamp_of_crash)
        """
# check how big this query will be
accident.estimate_query_size(query)
# store
hours_accident = accident.query_to_pandas_safe(query)
#Check head
hours_accident.tail()
# plot graph
import matplotlib as plt
hours_accident['f0_'].plot()
#Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state that were involved in
#hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016
#table for this, especially the registration_state_name and hit_and_run columns .
accident.table_schema('vehicle_2015')
accident.head("vehicle_2015")

query2 = """ SELECT COUNT(hit_and_run), registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run)
        """
accident.estimate_query_size(query2)
vehicle = accident.query_to_pandas_safe(query2)
vehicle.head()

vehicle.tail()
import matplotlib.pyplot as plt
vehicle.plot(x = 'registration_state_name', y = 'f0_')