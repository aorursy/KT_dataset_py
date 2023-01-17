# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.table_schema()
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
# Get table schema.
accidents.table_schema("accident_2015")

# Setup query.
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash),
                          COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# Esimate query size.
accidents.estimate_query_size(query)

# Query to dataframe.
df = accidents.query_to_pandas_safe(query)
print(df)
# Plot results.
my_plot = df.plot(kind = 'scatter', x = "f0_", y = "f1_")
my_plot.set_xlabel("Hour")
my_plot.set_ylabel("Fatalities")
my_plot.set_title("Most accidents are in the evening between 1700-2000")

# Get table schema.
accidents.table_schema("vehicle_2015")


# Setup query.
query = """SELECT registration_state_name, COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number) DESC
        """

# Estimate size of query.
accidents.estimate_query_size(query)

# Execute query.
df = accidents.query_to_pandas_safe(query)
print(df)
# Your code goes here :)

