# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import pandas as pd
import numpy as np # linear algebra
# google bigquery library for quering data
from google.cloud import bigquery
# BigQueryHelper for converting query result direct to dataframe
from bq_helper import BigQueryHelper
# matplotlib for plotting
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# import plotly
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
plotly.tools.set_config_file(world_readable=True, sharing='public')

%matplotlib inline
QUERY = """
    SELECT
        state_name,
        count(state_name) as state_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY state_name
    ORDER BY state_count ASC
        """
bq_assistant = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
df_state_popular = bq_assistant.query_to_pandas_safe(QUERY)

plt.subplots(figsize=(15,7))
sns.barplot(x='state_name',y='state_count',data=df_state_popular,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90, fontsize=20)
plt.xlabel('States', fontsize=20)
plt.title('Number of accidents in 2016 in different US states', fontsize=24)
plt.savefig('us_states.png')
plt.show()
QUERY = """
    SELECT
        light_condition_name,
        count(light_condition_name) as light_condition_name_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY light_condition_name
    ORDER BY light_condition_name_count ASC
        """
df_light_condition_popular = bq_assistant.query_to_pandas_safe(QUERY)


plt.subplots(figsize=(15,7))
sns.barplot(x='light_condition_name',y='light_condition_name_count',data=df_light_condition_popular,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90, fontsize=20)
plt.xlabel('Light condition', fontsize=20)
plt.title('Number of accidents in 2016 in different light conditions', fontsize=24)
plt.show()
QUERY = """
    SELECT
        atmospheric_conditions_1_name,
        count(atmospheric_conditions_1_name) as atmospheric_conditions_1_name_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY atmospheric_conditions_1_name
    ORDER BY atmospheric_conditions_1_name_count ASC
        """
df_atmosphoric_condition_popular = bq_assistant.query_to_pandas_safe(QUERY)


plt.subplots(figsize=(15,7))
sns.barplot(x='atmospheric_conditions_1_name',y='atmospheric_conditions_1_name_count',data=df_atmosphoric_condition_popular,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90, fontsize=20)
plt.xlabel('Atmospheric condition', fontsize=20)
plt.title('Number of accidents in 2016 in different atmospheric conditions', fontsize=24)
plt.show()

QUERY = """
    SELECT
        manner_of_collision_name,
        count(manner_of_collision_name) as manner_of_collision_name_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY manner_of_collision_name
    ORDER BY manner_of_collision_name_count ASC
        """
df_manner_collition_popular = bq_assistant.query_to_pandas_safe(QUERY)


plt.subplots(figsize=(15,10))
sns.barplot(y='manner_of_collision_name',x='manner_of_collision_name_count',data=df_manner_collition_popular,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90)
plt.yticks(fontsize=18)
plt.xlabel('Manner of collision', fontsize=20)
plt.title('The manner of collision of accidents in 2016', fontsize=24)
plt.show()

QUERY = """
    SELECT
        route_signing_name,
        count(route_signing_name) as route_signing_name_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY route_signing_name
    ORDER BY route_signing_name_count ASC
        """
df_route_signing_popular = bq_assistant.query_to_pandas_safe(QUERY)


plt.subplots(figsize=(15,10))
sns.barplot(y='route_signing_name_count',x='route_signing_name',data=df_route_signing_popular,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Route Sign', fontsize=20)
plt.title('Route sign name where accidents occur most', fontsize=24)
plt.show()

QUERY = """
    SELECT
        number_of_drunk_drivers,
        count(number_of_drunk_drivers) as number_of_drunk_drivers_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY number_of_drunk_drivers
    ORDER BY number_of_drunk_drivers_count ASC
        """
df_drak_drive_popular = bq_assistant.query_to_pandas_safe(QUERY)


plt.subplots(figsize=(15,10))
sns.barplot(y='number_of_drunk_drivers_count',x='number_of_drunk_drivers',data=df_drak_drive_popular,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Number of drunk driver', fontsize=20)
plt.title('The number of accidents occurs for drunk drivers', fontsize=24)
plt.show()

QUERY = """
    SELECT
        type_of_intersection,
        count(type_of_intersection) as type_of_intersection_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY type_of_intersection
    ORDER BY type_of_intersection_count ASC
        """
df_type_intersection_popular = bq_assistant.query_to_pandas_safe(QUERY)


plt.subplots(figsize=(15,10))
sns.barplot(y='type_of_intersection_count',x='type_of_intersection',data=df_type_intersection_popular,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Intersection names', fontsize=20)
plt.title('The number of accidents occurs in different intersections', fontsize=24)
plt.show()
QUERY = """
    SELECT
        land_use_name,
        count(land_use_name) as land_use_name_count
    FROM
      `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
    GROUP BY land_use_name
    ORDER BY land_use_name_count ASC
        """
df_land_popular = bq_assistant.query_to_pandas_safe(QUERY)


plt.subplots(figsize=(15,10))
sns.barplot(y='land_use_name_count',x='land_use_name',data=df_land_popular,palette='inferno',edgecolor=sns.color_palette('dark',7))
plt.ylabel('Number of accident', fontsize=20)
plt.xticks(rotation=90,fontsize=20)
plt.xlabel('Land', fontsize=20)
plt.title('The number of accidents occurs in lands', fontsize=24)
plt.show()
