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
accidents.head('accident_2015')
query = """SELECT COUNT(consecutive_number) as Count, 
                  EXTRACT(hour FROM timestamp_of_crash) as Hour_of_day
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY Hour_of_day
            ORDER BY COUNT(consecutive_number) DESC
        """
q1=accidents.query_to_pandas_safe(query)
import seaborn as sns
fig, ax = plt.subplots(figsize=(12,6)) 
sns.barplot(x='Hour_of_day',y='Count',data=q1)
plt.title('Hour of the Day with the most Accidents')
plt.show()
q1.to_csv('q1.csv')
q1
accidents.table_schema('vehicle_2015')
accidents.head('vehicle_2015',selected_columns=['registration_state_name','hit_and_run'])
query = """SELECT registration_state_name,count(*) as count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            where hit_and_run = 'Yes'
            GROUP BY registration_state_name            
            ORDER BY count desc
        """
q2 = accidents.query_to_pandas_safe(query)
q2
q2_top10 = q2.loc[1:10]
q2_top10.plot(x='registration_state_name',y='count',
                kind='bar',figsize=(15,6),title='States with the most hit and run',
                legend=False).set_ylabel('Count of Hit and Run Cases')
