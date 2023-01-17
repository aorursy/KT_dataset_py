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
accidents_by_day.head()
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# Your code goes here :)
# accidents in each hours of day in "accidents_2015" table
acc_hours_query = """SELECT COUNT(consecutive_number), 
                        EXTRACT(HOUR FROM timestamp_of_crash)
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                        GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                        ORDER BY COUNT(consecutive_number) DESC
                    """
acc_hours = accidents.query_to_pandas_safe(acc_hours_query)
#acc_hours[acc_hours['f1_'] == 13].head()
acc_hours.head()
acc_hours.to_csv("accidents_by_hours.csv")
hit_run_query = """SELECT registration_state_name,
                        COUNT(hit_and_run) as hit_run_count
                        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                        WHERE hit_and_run = 'Yes'
                        GROUP BY registration_state_name
                        ORDER BY hit_run_count DESC
                    """
hit_run = accidents.query_to_pandas_safe(hit_run_query)
hit_run.head(5)
hit_run.to_csv("hit_and_run_by_state.csv")