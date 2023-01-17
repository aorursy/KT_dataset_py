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
# Hours of the day that most accidents occur
import bq_helper

accident_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

#SQL qry
query_1 = """select hour_of_crash as crash_hour
            , count(timestamp_of_crash) as crashes
            from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            where hour_of_crash !=99
            group by hour_of_crash
            order by crash_hour asc

"""

accidents_by_hour = accident_data.query_to_pandas_safe(query_1)

#Query output:
accidents_by_hour
#Which State has the most hit and runs
query_2 = """select registration_state_name as state
                , count(consecutive_number) as hit_and_runs
            from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            where hit_and_run = 'Yes'
            group by registration_state_name
            order by hit_and_runs desc
"""
hit_and_run = accident_data.query_to_pandas_safe(query_2)

#Query output:
hit_and_run