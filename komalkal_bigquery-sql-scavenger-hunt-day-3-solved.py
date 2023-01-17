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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# print all the tables in this dataset 
accidents.list_tables()

# print information on all the columns in the "accident_2015" table
# in the accidents dataset
accidents.table_schema("accident_2015")

accidents.head("accident_2015")
#Question 1 - Which hours of the day do the most accidents occur during?

query_hours = '''SELECT COUNT(consecutive_number) as total_crash,
                  EXTRACT(HOUR FROM timestamp_of_crash) as hour_of_crash
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                  GROUP BY hour_of_crash
                  ORDER BY total_crash DESC 
                '''
accidents_by_hours = accidents.query_to_pandas_safe(query_hours)
display(accidents_by_hours)

#import matplotlib.pyplot as plt
#accidents_by_hours[['hour_of_crash']].plot.bar();

import seaborn as sns
sns.set_style("whitegrid")
ax = sns.barplot(x="hour_of_crash", y="total_crash", data=accidents_by_hours)
#Question 2 - Which state has the most hit and runs?

query_hitrun = """SELECT COUNT(consecutive_number) crash_id, 
                  registration_state_name,
                  hit_and_run,
                  COUNT(hit_and_run) as no_of_hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run in('yes', 'Yes', 'YES')
            GROUP BY registration_state_name, hit_and_run
            ORDER BY no_of_hit_and_run DESC
                      
        """
most_hit_and_run = accidents.query_to_pandas_safe(query_hitrun)
display(most_hit_and_run)
import seaborn as sns
sns.set_style("whitegrid")
ax = sns.barplot(x="no_of_hit_and_run", y="registration_state_name", data=most_hit_and_run.head(10))