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
# question 1
accidents.head("accident_2015")

my_query1 = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour = accidents.query_to_pandas_safe(my_query1)
accidents_by_hour
# question 2
accidents.head("vehicle_2015", selected_columns = ["hit_and_run", "registration_state_name", "consecutive_number"], num_rows=10)

my_query2 = """SELECT registration_state_name, COUNT(consecutive_number) AS num 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY num DESC
        """
hit_and_run_plates = accidents.query_to_pandas_safe(my_query2)
hit_and_run_plates
# extras - in which state hit_and_runs happen?
accidents.table_schema("accident_2015")
my_query3 = """SELECT a.state_name, COUNT(a.consecutive_number) AS num
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015` as v
            INNER JOIN `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015` as a
            ON v.consecutive_number = a.consecutive_number
            WHERE v.hit_and_run = 'Yes'
            GROUP BY state_name
            ORDER BY num DESC
        """
hit_and_run_states = accidents.query_to_pandas_safe(my_query3)
hit_and_run_states