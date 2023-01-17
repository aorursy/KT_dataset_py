#Loading Dataset
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
accidents_by_day = accidents.query_to_pandas_safe(query)
print(accidents_by_day)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")
#Which hours of the day do the most accidents occur during?

query_15="""SELECT COUNT(consecutive_number) as No_of_accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) as HOUR
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY HOUR
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour_2015 = accidents.query_to_pandas_safe(query_15)
print(accidents_by_hour_2015)
print("*******ANALYSIS*********")
print("Most of the accidents happen when the sun sets!!")


#2016 accidents analysis
query_16="""SELECT COUNT(consecutive_number) as No_of_accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) as HOUR
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY HOUR
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents_by_hour_2016 = accidents.query_to_pandas_safe(query_16)
print(accidents_by_hour_2016)
print("*******ANALYSIS*********")
print("Most of the accidents happen when the sun sets!!")



#Which state has the most hit and runs?
#Take a look at the dataset
accidents.head("vehicle_2015")


query="""SELECT COUNT(hit_and_run) as No_of_hit_and_runs, 
                  registration_state_name as State
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            where hit_and_run="Yes"
            GROUP BY State
            ORDER BY COUNT(hit_and_run) DESC
        """
no_of_hit_and_run_2015 = accidents.query_to_pandas_safe(query)
print(no_of_hit_and_run_2015)
print("*******ANALYSIS*********")
print("Most of the hit and run cases is reported in California[after the unknown value for the state]")
print("Unknown values can imply that the vehicle was unregistered or may be due to negligence the data was not recorded")
#For 2016
query="""SELECT COUNT(hit_and_run) as No_of_hit_and_runs, 
                  registration_state_name as State
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
              where hit_and_run="Yes"
            GROUP BY State
            ORDER BY COUNT(hit_and_run) DESC
        """
no_of_hit_and_run_2016 = accidents.query_to_pandas_safe(query)
print(no_of_hit_and_run_2016)
print("*******ANALYSIS*********")
print("Most of the hit and run cases is reported in California[after the unknown value for the state] even for the year 2016")

