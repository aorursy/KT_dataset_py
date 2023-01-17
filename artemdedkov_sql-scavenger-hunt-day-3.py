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
# Question 1: Which hours of the day do the most accidents occur during?
hours_query = """select count(*) Crashes, 
                        extract(hour from timestamp_of_crash) Hour
                from  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                group by Hour
                order by count(*) desc
                """
accidents_by_hour = accidents.query_to_pandas_safe(hours_query)
plt.plot(accidents_by_hour.Crashes, accidents_by_hour.Hour)
plt.title("Number of Accidents by Hour of a Day")
 # Question 2: Which state has the most hit and runs?
hnr_query = """select registration_state_name State,
                    countif(hit_and_run = 'Yes') No_hit_and_run
                from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                group by registration_state_name
                order by No_hit_and_run desc
            """

hnr_data = accidents.query_to_pandas_safe(hnr_query)
plt.figure(figsize=(18, 16))
plt.barh(hnr_data.State, hnr_data.No_hit_and_run)
plt.title("Number of hit and runs by state")
plt.xticks(fontsize = 18,rotation=90);
plt.yticks(fontsize = 16);