# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.head("accident_2015")
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
accidents.head("accident_2015")

# Hours of the Day with most accidents
query_hours = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query_hours)
accidents_by_hour
# The most accidents happen at 6pm, followed by 8pm and 5pm
# The least accidents happen at 4am, followed (surprisingly) by 8am and 3am

# Which state has the most drunk drivers involved in accidents (I don't have a hit and run column)
query_state = """SELECT state_name,SUM(number_of_drunk_drivers) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY state_name
            ORDER BY SUM(number_of_drunk_drivers) DESC
        """
drunk_drivers = accidents.query_to_pandas_safe(query_state)
drunk_drivers
# California has the most drunk drivers causing fatal accidents, followed by Texas and Florida
# Of course this is not quite fair to say, as we are neglecting the base rate of total fatal accidents here (which again relies on the number of drivers)