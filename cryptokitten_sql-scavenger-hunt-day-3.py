# Your code goes here :)
# Which hours of the day do the most accidents occur during?

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour

# Now lets sort the data by hour (starting with 0:00, i.e. ascending) and plot the results
import bq_helper
import matplotlib.pyplot as plt


# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash) as hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY hour
        """
accidents_by_hour = accidents.query_to_pandas_safe(query)
accidents_by_hour
# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Hour of the Day")
# Your code goes here :)
# Which state has the most hit and runs?

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(hit_and_run) as hit_and_run_count, 
                  registration_state_name as state
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """
hit_and_runs_by_state = accidents.query_to_pandas_safe(query)
hit_and_runs_by_state
