import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

from google.cloud import bigquery
from bq_helper import BigQueryHelper
db_traffic = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
query = """select hour_of_crash, count(consecutive_number)count_of_crashes
           from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
           group by hour_of_crash
           order by 2 desc """

crashes_per_hour = db_traffic.query_to_pandas(query)
x = crashes_per_hour['hour_of_crash']
y = crashes_per_hour['count_of_crashes']
y_pos = np.arange(len(y))

plt.bar(y_pos, y, align="center",alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('count of crashes')
plt.title('Count of crashes per hour')
plt.show()
query = """select 
                light_condition_name, 
                count(consecutive_number)count_of_crashes
           from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
           group by 
                light_condition_name
           order by 
               2 desc 
        """

crashes_per_light_condition = db_traffic.query_to_pandas(query)
x = crashes_per_light_condition['light_condition_name']
y = crashes_per_light_condition['count_of_crashes']
y_pos = np.arange(len(x))

plt.barh(y_pos, y, align="center",alpha=0.5)
plt.yticks(y_pos, x)
plt.xlabel('count of crashes')
plt.title('Count of crashes per light condition')
plt.show()

query = """select 
                speeding_related, 
                count(distinct a.consecutive_number)count_of_crashes
           from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` a
           join `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` b on a.consecutive_number = b.consecutive_number
           group by 
                speeding_related
           order by 
               2 desc 
        """

crashes_per_speeding_related = db_traffic.query_to_pandas(query)
x = crashes_per_speeding_related['speeding_related']
y = crashes_per_speeding_related['count_of_crashes']
y_pos = np.arange(len(x))

plt.barh(y_pos, y, align="center",alpha=0.5)
plt.yticks(y_pos, x)
plt.xlabel('count of crashes')
plt.title('Count of crashes per speeding related')
plt.show()

query = """select 
                roadway_surface_type, 
                count(distinct a.consecutive_number)count_of_crashes
           from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` a
           join `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` b on a.consecutive_number = b.consecutive_number
           where
               speeding_related = 'No'
           group by 
                roadway_surface_type
           order by 
               2 desc 
        """

crashes_per_roadway_surface_type= db_traffic.query_to_pandas(query)

x = crashes_per_roadway_surface_type['roadway_surface_type']
y = crashes_per_roadway_surface_type['count_of_crashes']
y_pos = np.arange(len(x))

plt.barh(y_pos, y, align="center",alpha=0.5)
plt.yticks(y_pos, x)
plt.xlabel('count of crashes')
plt.title('Count of crashes per roadway surface type')
plt.show()

query = """select 
                vehicle_make_name, 
                count(distinct a.consecutive_number)count_of_crashes
           from `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` a
           join `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` b on a.consecutive_number = b.consecutive_number
           where
               speeding_related = 'No'
           group by 
                vehicle_make_name
           having
                count(distinct a.consecutive_number)>500
           order by 
               2 desc 
        """

crashes_per_vehicle_make = db_traffic.query_to_pandas(query)

x = crashes_per_vehicle_make['vehicle_make_name']
y = crashes_per_vehicle_make['count_of_crashes']
y_pos = np.arange(len(x))

plt.barh(y_pos, y, align="center",alpha=0.5)
plt.yticks(y_pos, x)
plt.xlabel('count of crashes')
plt.title('Count of crashes per vehicle make')
plt.show()

query = """select 
                registration_state_name, 
                count(distinct a.consecutive_number)count_of_crashes
           from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` a
           group by 
                registration_state_name
           having
                count(distinct a.consecutive_number)>500
           order by 
               2 desc 
        """

crashes_per_state = db_traffic.query_to_pandas(query)
vehicles_per_state = pd.read_csv('../input/mv1.csv')

#crashes_per_state
relative_crashes_per_state = pd.merge(crashes_per_state, vehicles_per_state, left_on='registration_state_name', right_on='State')
relative_crashes_per_state["perc"] = relative_crashes_per_state["count_of_crashes"] / relative_crashes_per_state["Total"] * 100

#type(relative_crashes_per_state)
#relative_crashes_per_state.sort_values(by=['perc'], ascending=False)

relative_crashes_per_state_desc = relative_crashes_per_state.sort_values(by='perc', ascending=False)
x = relative_crashes_per_state_desc['registration_state_name']
y = relative_crashes_per_state_desc['perc']
y_pos = np.arange(len(x))

plt.barh(y_pos, y, align="center")
plt.yticks(y_pos, x)
plt.xlabel('% of crashes by total vehicles')
plt.title('Relative crashes per state')
plt.show()

relative_crashes_per_state_desc