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
accidents_by_day
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
print(accidents_by_day)
# display multiple print results on one line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
query1_2015 = """SELECT COUNT(consecutive_number) as Accident_Count, hour_of_crash 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour_of_crash
            ORDER BY COUNT(consecutive_number) DESC
        """
# load in a pandas df
accidents_by_hour = accidents.query_to_pandas_safe(query1_2015)
# The most accidents of the day occured at 6PM in 2015
# According to the Column Metadata "hour 99" refers to Not Notified or Unknown
accidents_by_hour.head()
df = accidents_by_hour.drop([24])
df.tail()
import seaborn as sns
% matplotlib inline
# plot 2015 accidents by hour
sns.set()
plt.figure(figsize=(10,7))
sns.barplot(x="hour_of_crash", y="Accident_Count", data=df)
plt.title("2015 Acccidents by Hour")
query1_2016 = """SELECT COUNT(consecutive_number) as Accident_Count, hour_of_crash 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY hour_of_crash
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour2 = accidents.query_to_pandas_safe(query1_2016)
# The most accidents of the day occured at 6PM in 2016
# This hour is the same as 2015
# If we look at the head of both years, most accidents occur during rush hour.
accidents_by_hour2.head()
# most hit and runs: hit_and_run | registration_state_name
query2_2015 = """SELECT COUNT(hit_and_run) as hnr_count, registration_state_name, hit_and_run
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                  GROUP BY registration_state_name, hit_and_run
                  ORDER BY COUNT(hit_and_run) DESC
              """

# display in pandas
hit_run = accidents.query_to_pandas_safe(query2_2015)
# print df
hit_run.head()
hit_run.tail()
hit_run.shape
# filter to see hit_and_run states that have values of Yes
yes = hit_run[hit_run['hit_and_run']=="Yes"]
yes
# plt hit and run
sns.set()
plt.figure(figsize=(5,10))
sns.barplot(x="hnr_count", y="registration_state_name", data=yes)
plt.title("2015 Hit and Run Accidents by State")
# most hit and runs: hit_and_run | registration_state_name
query2_2016 = """SELECT COUNT(hit_and_run) as hnr_count, registration_state_name, hit_and_run
                  FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                  GROUP BY registration_state_name, hit_and_run
                  ORDER BY COUNT(hit_and_run) DESC
              """
# display in pandas
hit_run2 = accidents.query_to_pandas_safe(query2_2016)
# print 2016
hit_run2.head()
hit_run2.tail()
hit_run2.shape
# filter to see hit_and_run states that have values of Yes
yes2 = hit_run2[hit_run2['hit_and_run']=="Yes"]
yes2.head()
# print to see the top 5 states, including unknown hit and run states
yes.head(6)
yes2.head(6)
