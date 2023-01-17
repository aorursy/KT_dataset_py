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
plt.plot(accidents_by_day.f_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)

import bq_helper
import matplotlib.pyplot as plt

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents_by_hour_query = """SELECT COUNT(consecutive_number),EXTRACT(HOUR from timestamp_of_crash)
                             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                             GROUP BY EXTRACT(HOUR from timestamp_of_crash)
                          """

accidents_by_hour_data = accidents.query_to_pandas_safe(accidents_by_hour_query)

#print(accidents_by_hour_data)

plt.plot(accidents_by_hour_data.f1_,accidents_by_hour_data.f0_,'ro')
plt.title("Accidents by hour")

import bq_helper
import matplotlib.pyplot as plt

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

accidents_hit_and_run_query = """SELECT registration_state_name, COUNT(hit_and_run)
                                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                 GROUP BY registration_state_name
                                 ORDER BY COUNT(hit_and_run)
                              """

#accidents_hit_and_run_data = accidents.query_to_pandas_safe(accidents_hit_and_run_query)
#accidents.table_schema("vehicle_2015")

accidents_hit_and_run_data = accidents.query_to_pandas_safe(accidents_hit_and_run_query)


accidents_hit_and_run_data.tail() 

#plt.plot(accidents_hit_and_run_data.registration_state_name, accidents_hit_and_run_data.f0_,'ro')
#plt.title("Hit and runs by state")

