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
# let's estimate a query size
accidents.estimate_query_size(query)
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day.head()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
plt.plot(accidents_by_day.sort_values(by='f1_').reset_index().f0_)
plt.title("Number of accidents by the day of week \n (where 0 - Monday and 6 - Sunday)")
query_H2016 = """SELECT COUNT(consecutive_number), 
                 EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
            """
accidents.estimate_query_size(query_H2016)
accidents_by_hours_2016 = accidents.query_to_pandas_safe(query_H2016)
accidents_by_hours_2016
plt.plot(accidents_by_hours_2016.sort_values(by='f1_').reset_index().f0_)
plt.title("Number of accidents by hour of the day in 2016")
accidents.head("vehicle_2016")
accidents.table_schema('vehicle_2016')
accidents.head('vehicle_2016', selected_columns='hit_and_run', num_rows=30)
query_hit_n_runs = """SELECT COUNT(consecutive_number),
                      registration_state_name
                      FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                      WHERE hit_and_run = "Yes"
                      GROUP BY registration_state_name
                      ORDER BY COUNT(consecutive_number) DESC
                   """
accidents.estimate_query_size(query_hit_n_runs)
hit_n_run_accidents_by_state = accidents.query_to_pandas_safe(query_hit_n_runs)
hit_n_run_accidents_by_state.head(10)