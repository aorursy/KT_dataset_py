# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash)
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt
# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Hour of Day \n (Most to least dangerous)")
query = """SELECT registration_state COUNT(vehicle_number), 
           FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
           GROUP BY registration_state
           HAVING (COUNT(vehicle_number) > 100
           ORDER BY COUNT(hit_and_run) DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
har_by_state = nhtsa_traffic_fatalities.query_to_pandas_safe(query)
print(har_by_state.head())
# library for plotting
import matplotlib.pyplot as plt
# make a plot to show that our data is, actually, sorted:
plt.plot(har_by_state.f0_)
plt.title("Number of Vehicles involved H+R by State \n (Most to least)")