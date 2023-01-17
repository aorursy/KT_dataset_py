# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head('accident_2015')
query_2015 = """ SELECT  COUNT(consecutive_number),
                         EXTRACT(HOUR FROM timestamp_of_crash)
                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                 GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                 ORDER BY COUNT(consecutive_number) DESC
             """ 

query_2016 = """ SELECT  COUNT(consecutive_number),
                         EXTRACT(HOUR FROM timestamp_of_crash)
                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                 GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
                 ORDER BY COUNT(consecutive_number) DESC
            """ 
accident_hours_2015 = accidents.query_to_pandas_safe(query_2015)
accident_hours_2016 = accidents.query_to_pandas_safe(query_2016)
accident_hours_2015.head(7)
accident_hours_2016.head(7)
sorted_for_plot_2015 = accident_hours_2015.sort_values('f1_')
sorted_for_plot_2016 = accident_hours_2016.sort_values('f1_')
import matplotlib.pyplot as plt

x = sorted_for_plot_2015.f1_
y = sorted_for_plot_2015.f0_

xx = sorted_for_plot_2016.f1_
yy = sorted_for_plot_2016.f0_

plt.figure(figsize=(10,5))
plt.plot(x,y, label='2015')
plt.plot(xx,yy, label='2016')
plt.legend()

plt.xlabel('hour of a day')
plt.title('number of accidents for each hour in a day \n in 2015 and 2016')

plt.grid()
plt.show()
accidents.head('vehicle_2015',3)
query = """ SELECT  registration_state_name,
                    COUNT(registration_state_name)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(registration_state_name) DESC
        """
state_vehicle = accidents.query_to_pandas_safe(query)
state_vehicle.head(7)
import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))
plt.plot(state_vehicle.registration_state_name, state_vehicle.f0_, marker = 'o')
plt.xticks(rotation=90)
plt.title('number of vehicles registered in each state that were involved in hit-and-run accidents')
plt.grid(color='b', linestyle='-', linewidth=0.5)
plt.show()
state_vehicle_dropped = state_vehicle.drop([0]) # lets drop unknown state names
import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))
plt.plot(state_vehicle_dropped.registration_state_name, state_vehicle_dropped.f0_, marker = 'o')
plt.xticks(rotation=90)
plt.title('number of vehicles registered in each state that were involved in hit-and-run accidents')
plt.grid(color='b', linestyle='-', linewidth=0.5)
plt.show()