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
#accidents.list_tables()
#accidents.head("accident_2015")
query = """
SELECT EXTRACT(HOUR FROM timestamp_of_crash),
COUNT(consecutive_number)   
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC
"""
accidents.query_to_pandas_safe(query)
#Didn't include non states since the instructions said to return vehicles registered in states
#Maybe this is too literal(?)
query="""
SELECT registration_state_name, COUNT(hit_and_run)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
WHERE hit_and_run="Yes" AND registration_state_name!="Unknown"
                        AND registration_state_name!="Not Applicable"
                        AND registration_state_name!="Other Foreign Country"
                        AND registration_state_name!="No Registration"
                        AND registration_state_name!="District of Columbia"
                        AND registration_state_name!="Other Registration (Includes Native American Indian Nations)"
GROUP BY registration_state_name
ORDER BY COUNT(hit_and_run) DESC
"""
accidents.query_to_pandas_safe(query)