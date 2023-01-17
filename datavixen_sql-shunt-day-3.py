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
# Return a table that has information on how many accidents occurred in each hour of the day in 2015, 
#sorted by the the number of accidents which occurred each hour. 
import pandas as pd
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="accident_2015")

#Need to figure this out.
query2 = """SELECT DATE(timestamp_of_crash),  
            EXTRACT(HOUR FROM timestamp_of_crash) AS `hour`,
            EXTRACT(DAY FROM timestamp_of_crash) AS `day`,
            EXTRACT(MONTH FROM timestamp_of_crash) AS `month`,
            COUNT(consecutive_number) AS `accidents`
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY timestamp_of_crash, month, day, hour
            ORDER BY accidents DESC
        """

accidents_by_month_day_hour = accidents.query_to_pandas_safe(query2)
df = accidents.query_to_pandas(query2)
df.groupby('hour').agg({"accidents":['sum', 'mean', 'var']})
query3 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS `hour`,
                  EXTRACT(DAY FROM timestamp_of_crash) AS `day`
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash), EXTRACT(DAY FROM timestamp_of_crash)
        """
accidents_by_hour_in_day = accidents.query_to_pandas_safe(query3)
accidents.estimate_query_size(query3)
print (accidents_by_hour_in_day)
df = accidents.query_to_pandas(query2)
df.head(3)
import bq_helper
import pandas as pd
nhtsa = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
nhtsa.list_tables()
nhtsa.head("vehicle_2015")
#nhtsa.table_schema("vehicle_2015") #use this to find out what type hit_and_run

#Return a table with the number of vehicles registered in each state 
#that were involved in hit-and-run accidents, 
#sorted by the number of hit and run

query4 = """SELECT registration_state_name, COUNT(hit_and_run)        
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes" 
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
         """
accidents_by_state = accidents.query_to_pandas_safe(query4)
df = accidents.query_to_pandas(query4)
df.head(5)
import matplotlib.pyplot as plt
from pylab import *
df_top = df[:11]
df_top = df_top[df_top.registration_state_name != 'Unknown']

state_name = df_top[['registration_state_name']]
y = df_top['f0_']
print(df_top)
pos = df_top.index.values   
figure(1)
barh(pos,y, align='center')    
yticks(pos, ('CA', 'FL','TX' ,'NY','MI','AZ','NC','WI','PN','GA'))
xlabel('Hit and Run')
grid(False)
show()
