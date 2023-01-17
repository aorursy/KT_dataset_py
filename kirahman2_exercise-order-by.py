# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to find out the number of accidents
# that happen every hour
query = """SELECT COUNT(consecutive_number),
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

# Creates data frame
accidents_by_hour = accidents.query_to_pandas_safe(query)

# library for plotting 
import matplotlib.pyplot as plt

# make a plot to show data is properly sorted
plt.plot(accidents_by_hour.f0_)
plt.title("Car Accidents By Hour \n (Most to least dangerous)")

# Print the first 5 lines of the data set
print(accidents_by_hour)
# import package with helper function
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to find out the number of hit and run 
# accidents that occur in each state in descending order
query = """SELECT registration_state_name, COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
hit_and_run_by_state = accidents.query_to_pandas_safe(query)

# library for plotting
import matplotlib.pyplot as plt

# makes a plot to show that the data is sorted
plt.plot(hit_and_run_by_state.f0_)
plt.title("Number of Hit and Run Accidents by State")

# print head of dataset
print(hit_and_run_by_state)

