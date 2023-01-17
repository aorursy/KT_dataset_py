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
#Which hours of the day to most accidents occur
#Return table with hour of the day in 2015

#Setup Big Query helper object
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")


accidents.head("accident_2015")


#Setup Query

hour_query = """
Select count(state_number), EXTRACT (HOUR from timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY extract(HOUR from timestamp_of_crash)
ORDER BY count(state_number) Desc

"""



#Execute Query
accidents_by_hour = accidents.query_to_pandas_safe(hour_query)
#Clense and check data
accidents_by_hour.columns = ['quantity','hour']
accidents_by_hour.set_index('hour',inplace=True)
accidents_by_hour.sort_index(inplace=True)
accidents_by_hour.head()

import matplotlib.pyplot as plt
#PLot including a median line
accidents_by_hour.plot.bar(figsize=(20,10))
plt.axhline(accidents_by_hour['quantity'].median())
#Reuse the query - look at the data
accidents.head("vehicle_2015")

#Note cant see the datas sources - need to look at the data dictionary for details
state_query = """

Select count(hit_and_run), registration_state_name
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
where hit_and_run = "Yes"
group by registration_state_name
order by registration_state_name

"""
#Execute Query
hit_run_state = accidents.query_to_pandas_safe(state_query)
#Set state to index
hit_run_state.set_index('registration_state_name',inplace=True)

#Rename quantity
hit_run_state.columns = ['quantity']

#Sort by state ascending
hit_run_state.sort_values('quantity',inplace=True,ascending=False)

#Remove unknown
hit_run_state = hit_run_state[hit_run_state.index != "Unknown"]
hit_run_state.head()
#Graph the top 20 states - any more wouldn't be readable
ax = hit_run_state[:20].plot.bar(figsize=(20,10))
ax.set_xlabel('')
fig=plt.gcf()
plt.xticks(fontsize=18, rotation=75)
