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
accidents.list_tables()
accidents.head('accident_2015')
accidents.table_schema('accident_2015')
query1 = """SELECT COUNT(consecutive_number),
                   EXTRACT(HOUR from timestamp_of_crash)
                   FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                   GROUP BY EXTRACT(HOUR from timestamp_of_crash)
                   ORDER BY COUNT(consecutive_number) DESC
        """
accidents.estimate_query_size(query1)
accidents_by_hour = accidents.query_to_pandas_safe(query1)
import matplotlib.pyplot as plt

plt.plot(accidents_by_hour.f0_)
plt.title("Number of Accidents by Rank of Hour \n (Most to least dangerous)")
plt.xlabel('Order')
plt.ylabel('Number of Accidents')

#arrow: xy=(,) is the end point of the arrow and xytext=(,) is the start point of the text
plt.annotate('13h: 1387 accidents', xy=(10, 1387), xytext=(15,1600),
            arrowprops=dict(facecolor='b', shrink = 0.1),
            )
plt.minorticks_on() #Display minor ticks on the current plot.
print(accidents_by_hour)
accidents.head(
    "vehicle_2015", 
    selected_columns = ["hit_and_run", "registration_state_name", "consecutive_number"],
    num_rows = 10
)
query2 = """SELECT registration_state_name,
                   COUNT(hit_and_run)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
         """
accidents.estimate_query_size(query2)
hitnrun = accidents.query_to_pandas(query2)
hitnrun
import matplotlib.pyplot as plt

plt.plot(hitnrun.f0_)
plt.title('Number of accidents ranked by \n Registration state name')
plt.xlabel('The ID of the State')
plt.ylabel('Nuber of accidents')
