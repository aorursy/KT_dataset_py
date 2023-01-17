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
accidents_by_day
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
accidents.list_tables()
accidents.head("accident_2015")
query = '''
        SELECT EXTRACT(HOUR FROM timestamp_of_crash),
               COUNT(*)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY 1 
        ORDER BY 2 DESC
        '''
accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f1_)
plt.title("Number of Accidents by Rank of HOUR \n (Most to least dangerous)")
accidents.head("vehicle_2016", num_rows=2)
accidents.table_schema("vehicle_2016") #to check the complete schema
query = '''
        SELECT registration_state_name ,
               COUNT(hit_and_run)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
        WHERE hit_and_run = 'Yes'
        GROUP BY 1 
        ORDER BY 2 DESC
        '''

hit_and_runs_by_state_name = accidents.query_to_pandas_safe(query)
hit_and_runs_by_state_name
import matplotlib.pyplot as plt
import seaborn as sns

ax = plt.subplots(figsize=(15,7))
sns.set_style('darkgrid')
ax = sns.barplot(x='f0_', y='registration_state_name', data=hit_and_runs_by_state_name, palette='Blues_r')
