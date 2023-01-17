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
accidents.list_tables()
accidents.head('accident_2015')
query1 = """ select count(consecutive_number),
             EXTRACT(HOUR FROM timestamp_of_crash) FROM
             `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` 
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) """
accidents.estimate_query_size(query1)
res = accidents.query_to_pandas_safe(query1)
res.head()

import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(res.f0_)
plt.title("Number of Accidents by Rank of HOUR \n (least to most dangerous)")
accidents.table_schema('vehicle_2015')
accidents.head('vehicle_2016')
query2 = """SELECT state_number,count(consecutive_number)
        from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        group by state_number order by count(consecutive_number) DESC """
accidents.estimate_query_size(query2)
res = accidents.query_to_pandas_safe(query2)
res.head()