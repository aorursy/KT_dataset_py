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
%matplotlib inline

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

# Print the .head() of the "accident_2016" table
accidents.head("accident_2016")
# Setting the QUERYs

QUERY1 = """SELECT COUNT(consecutive_number) AS accidents_count, EXTRACT(HOUR FROM timestamp_of_crash) AS hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hour
            ORDER BY COUNT(consecutive_number) DESC
            """

QUERY2 = """SELECT registration_state_name AS state, COUNT(hit_and_run) AS hit_and_run
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY state
            ORDER BY hit_and_run DESC
            """
# Solving question 1
accidents_hour = accidents.query_to_pandas_safe(QUERY1)
# Checking our query result
accidents_hour
# Plotting accidents ordered by hour
# Experimenting using arrowprops in plt.annotate()

# Changing hour type to categoy
accidents_hour.hour = accidents_hour.hour.astype('category')

# Sort hour column and plotting the graph
accidents_hour.sort_values(by='hour').plot(x='hour', y='accidents_count', figsize=(12, 8));

# Finding the max point x and y coordinates 
max_accident_y = accidents_hour.accidents_count.max()
max_accident_x = accidents_hour.hour[accidents_hour.accidents_count.argmax()]

# Annotating with the most dangerous hour information
plt.annotate('Most accidents at %d hours' % max_accident_x, 
             xy=(max_accident_x-0.3, max_accident_y-10), 
             xytext=(max_accident_x-10, max_accident_y-200), 
             arrowprops=dict(facecolor='black'));
# Solving question 2
hit_and_run = accidents.query_to_pandas_safe(QUERY2)
# Checking our query result
hit_and_run.head()
# Plotting the first 5 states with most hit and run cases
hit_and_run[1:6].plot(kind='bar', x='state', figsize=(12, 8));
plt.xticks(rotation=60);
plt.ylabel('Hit and run count');
plt.title('Hit and run by state');