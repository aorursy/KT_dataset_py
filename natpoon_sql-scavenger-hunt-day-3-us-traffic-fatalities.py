# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# Define the query
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
# Assign a variable name to the query (type: dataframe)
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
accidents.list_tables()
accidents.head('accident_2015')
# Define query2
query2 = """SELECT COUNT(consecutive_number),
                   EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
            """
accidents_by_hour = accidents.query_to_pandas_safe(query2)
accidents_by_hour
accidents_by_hour.to_csv("accidents_by_hour.csv")
# Plotting number of accidents by hour of day
import seaborn as sns
sns.set_style("whitegrid") 
ax = sns.barplot(x="f1_", y="f0_", data=accidents_by_hour)
ax.set(xlabel='Hour of Day', ylabel='Number of Accidents')
# Your code goes here :)
query3 = """SELECT registration_state_name,
                   COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
         """

hitruns_by_state_2015 = accidents.query_to_pandas_safe(query3)
hitruns_by_state_2015
hitruns_by_state_2015.to_csv("hitruns_by_state_2015.csv")
query4 = """SELECT registration_state_name,
                   COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            WHERE hit_and_run = "Yes"
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
         """

hitruns_by_state_2016 = accidents.query_to_pandas_safe(query4)
hitruns_by_state_2016
hitruns_by_state_2016.to_csv("hitruns_by_state_2016.csv")
