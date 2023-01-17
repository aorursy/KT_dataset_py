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
#accidents.table_schema(table_name="accident_2015") #uncomment for more info
accidents.head(table_name="accident_2015")
accidents.head(table_name="accident_2015", selected_columns=["consecutive_number", "timestamp_of_crash"])
#Which hours of the day do the most accidents occur during?

query1 = """SELECT COUNT(consecutive_number), EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
         """

accidents_by_hour_df = accidents.query_to_pandas_safe(query=query1)

accidents_by_hour_df.head(n=24)
plt.plot(accidents_by_hour_df.f0_)
plt.title(s="Number of accidents by Rank of hour \n (Most to least dangerous)")
#Which state has the most hit and runs?
#accidents.table_schema(table_name="vehicle_2015") #uncomment for more info
accidents.head(table_name="vehicle_2015", 
               selected_columns=["consecutive_number", "registration_state_name", "hit_and_run"], 
               num_rows=30)
query2 = """SELECT COUNT(hit_and_run), registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
         """

hit_and_run_statewise_df = accidents.query_to_pandas_safe(query=query2)

hit_and_run_statewise_df.head(len(hit_and_run_statewise_df["f0_"]))