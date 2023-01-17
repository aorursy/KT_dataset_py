# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.table_schema('accident_2015')
accidents.head('accident_2015')
# query to find out the number of accidents which 
# happen on each hour of the day
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash), 
            COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number)
        """
accidents.estimate_query_size(query)
accidents_by_hour = accidents.query_to_pandas_safe(query)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:

plt.plot(accidents_by_hour.f1_)
plt.title("Number of Accidents by Hour Day ranked by number of accidents\n ")
#plt.bar(accidents_by_hour.f0_, accidents_by_hour.f1_)
#plt.show()

print(accidents_by_hour)
accidents.head('vehicle_2015', selected_columns='registration_state_name,hit_and_run')
# query to find out the number of accidents by state
# ordered by number of accidents
query = """SELECT registration_state_name, 
            COUNT(consecutive_number)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(consecutive_number)
        """
accidents.estimate_query_size(query)
no_accidents_state = accidents.query_to_pandas_safe(query)
print(no_accidents_state)
import matplotlib.pyplot as plt

plt.plot(no_accidents_state.f0_)
plt.title('Number of accidents per state')