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

# print the accident_2015 table structure
accidents.table_schema("accident_2015")

# print few records from "accident_2015" table
accidents.head("accident_2015")

# query to find out how many accidents happen hourly in 2015
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash),
                  COUNT(consecutive_number)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
        """
# estimate query size
accidents.estimate_query_size(query)

# as the query size is little we run it directly
accidents_by_hr = accidents.query_to_pandas(query)

# lets check the dataframe for the sorted result returned by the query
accidents_by_hr

# we can create the plot as given in the example above
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hr.f1_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least)")

# second question of the hunt
# check the list of tables in the dataset
accidents.list_tables()

# we are going to work with vehicle_2015 table, print the structure
accidents.table_schema("vehicle_2015")

# lets check few records to understand the data
accidents.head("vehicle_2015")

# The columns we are interested is not showing up
# lets print few selected columns of our interest
accidents.head("vehicle_2015", selected_columns="registration_state,hit_and_run,registration_state_name", num_rows=10)

# Above is good but not giving us a clear picture on the different values of hit_and_run
# print out different values in hit_and_run to better understand the column values
query = """SELECT DISTINCT hit_and_run
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
        """

# first estimate query size
accidents.estimate_query_size(query)

# only 0.19MB required, lets run the query
accidents.query_to_pandas(query)

# we have Yes/No/Unknown values and only interested for Yes to answer the question
query = """SELECT registration_state_name,COUNTIF(hit_and_run = "Yes")
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name 
            ORDER BY COUNTIF(hit_and_run = "Yes") DESC
        """
# first estimate query size
accidents.estimate_query_size(query)

# only 0.68MB required, lets run the query and store the result in a dataframe
state_wise_hit = accidents.query_to_pandas(query)

# print the result out
print(state_wise_hit)

# we can see that State of California is having highest Hit and Run cases 