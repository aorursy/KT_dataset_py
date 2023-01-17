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
# Which hours of the day do the most accidents occur during?



query = """ select extract(hour from timestamp_of_crash) as accident_hour, count(consecutive_number) as accident_counts

from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

group by accident_hour

order by accident_counts DESC

 """ 





accidents_by_hour = accidents.query_to_pandas_safe(query)

accidents_by_hour
# library for plotting

import matplotlib.pyplot as plt

fig, ax = plt.subplots()



# bar plot according to hour of the accident

plt.bar(accidents_by_hour['accident_hour'], accidents_by_hour['accident_counts'],align='center', alpha=0.5)

plt.title("Number of Accidents in each hour")





print('Wee see that at 6pm has the highest accidents and the number of accidentst happens the most in the evening time after 3pm')
# Which state has the most hit and runs?



## we need to use VEHICLE_2015 table to find number of hit & run cases



query = """ select registration_state_name as state, count(hit_and_run) as count_hit_run 

from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`

where hit_and_run = 'Yes'

group by state

order by count_hit_run desc"""



hit_run = accidents.query_to_pandas_safe(query)



print('-> It is very difficult to say which state has maximum hit_run cases as it is Unknown')

print('-> California being the second highest among other states with 155 hit_run cases')

hit_run


