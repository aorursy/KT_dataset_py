# Libraries
from google.cloud import bigquery
import pandas as pd
from bq_helper import BigQueryHelper

# BigQuery client
client = bigquery.Client()

# Makes bq_assistant object
bq_assistant_2015 = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities.accident_2015")

# accident_2015 query
query_2015 = """
#standardSQL
SELECT state_name, number_of_motor_vehicles_in_transport_mvit,
number_of_persons_not_in_motor_vehicles_in_transport_mvit,
number_of_persons_in_motor_vehicles_in_transport_mvit, county, city,
day_of_crash, month_of_crash, year_of_crash, day_of_week, hour_of_crash,
minute_of_crash, functional_system_name, first_harmful_event_name, manner_of_collision_name,
light_condition_name, atmospheric_conditions_name, atmospheric_conditions_1_name,
atmospheric_conditions_2_name, number_of_fatalities, hour_of_notification,
minute_of_notification, hour_of_arrival_at_scene, minute_of_arrival_at_scene, hour_of_ems_arrival_at_hospital,
minute_of_ems_arrival_at_hospital, COUNT(*) AS count
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY state_name, number_of_motor_vehicles_in_transport_mvit,
number_of_persons_not_in_motor_vehicles_in_transport_mvit,
number_of_persons_in_motor_vehicles_in_transport_mvit, county, city,
day_of_crash, month_of_crash, year_of_crash, day_of_week, hour_of_crash,
minute_of_crash, functional_system_name, first_harmful_event_name, manner_of_collision_name,
light_condition_name, atmospheric_conditions_name, atmospheric_conditions_1_name,
atmospheric_conditions_2_name, number_of_fatalities, hour_of_notification,
minute_of_notification, hour_of_arrival_at_scene, minute_of_arrival_at_scene, hour_of_ems_arrival_at_hospital,
minute_of_ems_arrival_at_hospital
"""

df_2015 = bq_assistant_2015.query_to_pandas_safe(query_2015)

# Makes bq_assistant object
bq_assistant_2016 = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities.accident_2016")

# accident_2016 query
query_2016 = """
#standardSQL
SELECT state_name, number_of_motor_vehicles_in_transport_mvit,
number_of_persons_not_in_motor_vehicles_in_transport_mvit,
number_of_persons_in_motor_vehicles_in_transport_mvit, county, city,
day_of_crash, month_of_crash, year_of_crash, day_of_week, hour_of_crash,
minute_of_crash, functional_system_name, first_harmful_event_name, manner_of_collision_name,
light_condition_name, atmospheric_conditions_name, atmospheric_conditions_1_name,
atmospheric_conditions_2_name, number_of_fatalities, hour_of_notification,
minute_of_notification, hour_of_arrival_at_scene, minute_of_arrival_at_scene, hour_of_ems_arrival_at_hospital,
minute_of_ems_arrival_at_hospital, COUNT(*) AS count
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY state_name, number_of_motor_vehicles_in_transport_mvit,
number_of_persons_not_in_motor_vehicles_in_transport_mvit,
number_of_persons_in_motor_vehicles_in_transport_mvit, county, city,
day_of_crash, month_of_crash, year_of_crash, day_of_week, hour_of_crash,
minute_of_crash, functional_system_name, first_harmful_event_name, manner_of_collision_name,
light_condition_name, atmospheric_conditions_name, atmospheric_conditions_1_name,
atmospheric_conditions_2_name, number_of_fatalities, hour_of_notification,
minute_of_notification, hour_of_arrival_at_scene, minute_of_arrival_at_scene, hour_of_ems_arrival_at_hospital,
minute_of_ems_arrival_at_hospital
"""

df_2016 = bq_assistant_2016.query_to_pandas_safe(query_2016)

# Stacks data and writes output
df = df_2015.append(df_2016)

df.to_csv("nhtsa.csv")
