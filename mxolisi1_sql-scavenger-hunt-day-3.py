# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

#** print the first couple rows of the "accident_2016" table
accidents.head("accident_2016")
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number) as accidents
             , 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
#* check howlong it'll charge you( i.e. how BIG it is)

accidents.estimate_query_size(query)#0.0004848688840866089(i..e. 0.4MBs)*
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)

#              #* you may view the first 7 rows
accidents_by_day.head(7)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.accidents)#*plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Your code goes here :)
#**Which hours of the day do the most accidents occur during?
#        *Return a table that has information on how many accidents occurred in each hour of the day in 2015,
#sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or 
#  accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column,
#but if you use that one you won't get a chance to practice with dates. :P)
#        *Hint: You will probably want to use the EXTRACT() function for this.
########
#              #* you may view the first 7 rows again
accidents.head("accident_2016")#;accidents.head("accident_2015")

#              #* the query
# query to find out  how many accidents occurred in each hour of the day in 2015
query = """SELECT COUNT(consecutive_number) as accidents
             , 
                  EXTRACT(HOUR FROM timestamp_of_crash) 
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
#* check howlong it'll charge you( i.e. how BIG it is)

accidents.estimate_query_size(query)#0.0004848688840866089(i..e. 0.4MBs)*
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_per_hour = accidents.query_to_pandas_safe(query)

#      #*view the head
accidents_per_hour.head(7)#print(accidents_per_hour)
# library for plotting
import matplotlib.pyplot as plt
import seaborn as sns
# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_per_hour.accidents)#*plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
#* got this from user(Gabe) NB** 
sns.set_style("whitegrid") 
ax = sns.barplot(x="f0_", y="accidents", data=accidents_per_hour,palette='coolwarm')

#* add title and label tothe x-axis
plt.title("Number of Accidents Hour \n (Higher peaks & late hours are 'redish')")
# Set x-axis label
plt.xlabel('Hour')
#    Which state has the most hit and runs?
#    *Return a table with the number of vehicles registered in each state that were involved in
#hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or 
     #vehicle_2016 table for this, especially the registration_state_name and
#hit_and_run columns.Which state has the most hit and runs?


########

#              #* you may view the first 7 rows again
accidents.head("accident_2016")#;accidents.head("accident_2015")

accidents.head("vehicle_2016")#;
accidents.head("vehicle_2015")
#              #* the query
# query to find out  how many accidents occurred in each hour of the day in 2015
query2 = """SELECT COUNT(hit_and_run) as hitnruns
                   , registration_state_name
                  
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                  Where hit_and_run='Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
#* check howlong it'll charge you( i.e. how BIG it is)

accidents.estimate_query_size(query2)#0.0004848688840866089(i..e. 0.5MBs)*
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_run_per_state = accidents.query_to_pandas_safe(query2)

#      #*view the head
#print(hit_run_per_state)
hit_run_per_state.head()
f, ax = plt.subplots(figsize=(6, 15))
sns.set_style("whitegrid") 
ax = sns.barplot(x="hitnruns", y="registration_state_name", data= hit_run_per_state,palette='coolwarm',dodge=False)

#*add a title & label the axes!
plt.title("Number of Hit and Runs per State")
# Set x-axis label
plt.xlabel('Hit & Runs')
# Set x-axis label
plt.ylabel('State')