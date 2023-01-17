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
#Beginning or first query 
query1 = """SELECT EXTRACT(HOUR from timestamp_of_crash) as hour, COUNT(consecutive_number) as number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY number DESC"""
#Making data frame
accidents_by_hour = accidents.query_to_pandas_safe(query1)
accidents_by_hour
#Computing average time for crashes
accidents_by_hour["prod"] = accidents_by_hour["hour"]*accidents_by_hour["number"]
x = accidents_by_hour["number"].sum()
mean = accidents_by_hour["prod"].sum()/x
#We can build some graphs
plt.bar(accidents_by_hour.hour, accidents_by_hour.number)
plt.axvline(mean, color = 'r')
plt.title("Number of accidents in the diferent hours\n (red line is mean)")
plt.show()
#Buiding query
query2 = """SELECT registration_state_name as state_name, COUNT(hit_and_run) as number
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY state_name
            ORDER BY number DESC
            """
states = accidents.query_to_pandas_safe(query2)
states
#Well Unknow state is not very important so we will drop it
states.drop(states.index[0], inplace = True)
states
#And pieplot for states with biggest numbers
states2 = states.head(n = 10)
plt.figure(figsize = (9,9))
plt.pie(states2.number, labels = states2.state_name)
plt.show()