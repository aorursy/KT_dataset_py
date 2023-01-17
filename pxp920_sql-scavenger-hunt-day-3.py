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
# which hours of the day do the most accidents occur during?

query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_day = accidents.query_to_pandas_safe(query)
# Look at the fist few records of accidents 2006 table
accidents.head("accident_2016")
# which hours of the day do the most accidents occur during?
# I count the number of accidents by hour of the day for 2016 below
# NOTE: It appears that I'm "forced" to use the aliases in the Group By statement, otherwise it results
# in an error
query = """SELECT EXTRACT(hour FROM timestamp_of_crash) as HourofDay,
            COUNT(*) as NumberOfAccidents        
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY HourofDay
            ORDER BY HourofDay asc
        """
accidents.query_to_pandas_safe(query)
# Look at the fist few records of vehicles 2006 table
accidents.head("vehicle_2016")
# Check available categories for "hit_and_run" column
query = """ SELECT distinct hit_and_run     
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        """
accidents.query_to_pandas_safe(query)
# Which state has the most hit and runs?
# First subquery to return the state number and state name mapping
# Second subquery to provide the count of hit and runs by state
# Query to join the two and return overall results by descending hit and run order
query = """
            select state_name, a.state_number, NumberOfHitNRuns
            from 
            (
                SELECT distinct state_name, state_number      
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            ) a
            join
            (
                SELECT state_number, count(*) as NumberOfHitNRuns
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                where hit_and_run="Yes"
                group by state_number
            ) b
            on a.state_number = b.state_number
            order by NumberOfHitNRuns desc
        """
results = accidents.query_to_pandas_safe(query)
results[0:5]
# A quick visual - easier to digest
import matplotlib.pyplot as pl
import numpy as np
plt.figure(num=None, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
height = results.NumberOfHitNRuns
bars = results.state_name
y_pos = np.arange(len(bars))
plt.bar(y_pos, height, align='center')

plt.xticks(y_pos, bars, color='black', rotation=90, fontsize='6')
plt.tick_params(labelbottom='on')
pl.title("Number of Hit and Runs by State \n (Descending Order)")