# Importing Libraries
import bq_helper 
import numpy as np 
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from statsmodels.graphics.mosaicplot import mosaic

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# List of available tables
dataset= bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="nhtsa_traffic_fatalities")
dataset.list_tables()
# Head of tables:
dataset.head('vehicle_2016',3)
dataset.head('pbtype_2016',3)
dataset.head('accident_2016',3)
# Left join of "vehicle_2016" and "pbtype_2016"
SQL_join_vehicle_pbtype = """
SELECT  
v.consecutive_number, v.number_of_occupants, v.travel_speed, v.speed_limit, v.vehicle_model_year, 
v.month_of_crash, v.hour_of_crash, v.previous_recorded_crashes, v.previous_speeding_convictions,
p.sex, p.age
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` as v
LEFT JOIN `bigquery-public-data.nhtsa_traffic_fatalities.pbtype_2016` as p
ON v.consecutive_number = p.consecutive_number
"""
# Create and save set1 as Panda's dataframe: 
dataset.estimate_query_size(SQL_join_vehicle_pbtype)
set1= dataset.query_to_pandas_safe(SQL_join_vehicle_pbtype)
set1.to_csv("join_vehicle_pbtype_2016.csv")
# first and last 5 observations:
set1.head()
set1.tail()
# More information:
set1.info()
# Change type of "month_of_crash" and "hour_of_crash":
set1["month_of_crash"]= set1["month_of_crash"].astype("category")
set1["hour_of_crash"]= set1["hour_of_crash"].astype("category")
# Plot effect of "month_of_crash"
set1["month_of_crash"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("month")
plt.ylabel("Number of accidents")
plt.show()
# Plot effect of "hour_of_crash"
set1["hour_of_crash"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("hour")
plt.ylabel("Number of accidents")
plt.show()
# Change type of "vehicle_model_year" 
set1["vehicle_model_year"]= set1["vehicle_model_year"].astype("category")
# Plot effect of "vehicle_model_year"
plt.figure(figsize=(12,4))
set1["vehicle_model_year"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("vehicle model")
plt.ylabel("Number of accidents")
plt.title("Barplot of number of accidents vs vehicle model")
plt.show()
# Zoom in after 1990:
set1["vehicle_model_year"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("vehicle model")
plt.ylabel("Number of accidents")
plt.axis(xmin=51)
plt.title("Barplot of number of accidents vs vehicle model (after 1990)")
plt.show()
# Plot "previous_recorded_crashes" vs "Number of accidents":
set1["previous_recorded_crashes"].value_counts().sort_index()[0:10].plot(kind="bar")
plt.xlabel("previous recorded crashes")
plt.ylabel("Number of accidents")
plt.show()
# Plot "previous_speeding_convictions" vs "Number of accidents":
set1["previous_speeding_convictions"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("previous speeding convictions")
plt.ylabel("Number of accidents")
plt.show()
# Change type of "sex" 
set1["sex"]= set1["sex"].astype("category")
# Plot "sex" vs "Number of accidents":
set1["sex"].value_counts().plot(kind="pie")
plt.ylabel("Number of accidents")
plt.xlabel("gender")
plt.show()
# Plot "age " vs "Number of accidents":
set1[set1["age"]<200]["age"].plot(kind="box")
plt.xlabel("driver age")
plt.ylabel("Number of accidents")
plt.show()
# Plot "travel_speed" vs "Number of accidents":
set1[set1["travel_speed"]<200]["travel_speed"].plot.hist(bins=15)
plt.xlabel("speed")
plt.ylabel("Number of accidents")
plt.show()
# Plot "speed_limit" vs "Number of accidents":
set1["speed_limit"].plot(kind="hist")
plt.xlabel("speed limit")
plt.ylabel("Number of accidents")
plt.show()
# Rate of fatality and drunk driver of US state:
SQL_accident_rate = """
SELECT state_name, 
SUM(latitude)/COUNT(consecutive_number) as avg_lat, 
SUM(longitude)/COUNT(consecutive_number) as avg_lon, 
COUNT(consecutive_number) as accidents,   
SUM(number_of_fatalities) as fatality, 
SUM(number_of_drunk_drivers) as drunk_driver, 
SUM(number_of_fatalities)/COUNT(consecutive_number) as fatality_rate, 
SUM(number_of_drunk_drivers)/COUNT(consecutive_number) as drunk_rate
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` 
GROUP BY state_name
ORDER BY state_name

"""
# Create and save set2 as Panda's dataframe: 
dataset.estimate_query_size(SQL_accident_rate)
set2= dataset.query_to_pandas_safe(SQL_accident_rate)
set2.to_csv("accident_rate_2016.csv")
# See result of above query:
set2
# More information:
set2.info()
# Add Abb of states
us_state_abb = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
                'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 
                'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI',
                'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 
                'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME',
                'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 
                'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
                'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 
                'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 
                'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 
                'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
                'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 
                'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 
                'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 
                'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 
                'Dist of Columbia': 'DC'}
# Adding codes column to the dataframe:
set2['state_abb'] = set2['state_name'].map(us_state_abb)
set2
plt.figure(figsize=(12,8))
Map = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130.,
              urcrnrlon=-60.,lat_ts=20,resolution='i')
Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()
x,y = Map(set2['avg_lon'].values, set2['avg_lat'].values) #transform to projection
d_size= (np.around(500*set2['drunk_rate'].values.astype("float"))).tolist()
Map.scatter(x,y, c="b", s=d_size)
plt.title("drunk_rate of states")
plt.show()
plt.figure(figsize=(12,8))
Map = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130.,
              urcrnrlon=-60.,lat_ts=20,resolution='i')
Map.drawcoastlines()
Map.drawcountries()
Map.drawstates()
x,y = Map(set2['avg_lon'].values, set2['avg_lat'].values) #transform to projection
f_size= (np.around(500*set2['fatality_rate'].values.astype("float"))).tolist()
Map.scatter(x,y, c="b", s=f_size)
plt.title("fatality_rate of states")
plt.show()
# Light and atmospheric conditions effect:
SQL_accident_weather = """
SELECT
light_condition_name, atmospheric_conditions_1_name,
COUNT(consecutive_number) as accidents,   
SUM(number_of_fatalities) as fatality
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` 
GROUP BY atmospheric_conditions_1_name, light_condition_name
ORDER BY atmospheric_conditions_1_name, light_condition_name

"""
# Create and save set3 as Panda's dataframe: 
dataset.estimate_query_size(SQL_accident_weather)
set3= dataset.query_to_pandas_safe(SQL_accident_weather)
set3.to_csv("accident_weather_2016.csv")
set3
# Type of variables:
set3.info()
# Contengenccy table showing number of accidents:
weather_accident= set3.iloc[:, 0:3].pivot_table(index="light_condition_name", columns="atmospheric_conditions_1_name", values="accidents", aggfunc=np.sum)
weather_accident
# Remove "Not Reported", "Other", "Unkown" from dataset:
set3=set3.drop(set3.loc[set3["atmospheric_conditions_1_name"]=="Not Reported"].index)
set3=set3.drop(set3.loc[set3["atmospheric_conditions_1_name"]=="Other"].index)
set3=set3.drop(set3.loc[set3["atmospheric_conditions_1_name"]=="Unknown"].index)

set3=set3.drop(set3.loc[set3["light_condition_name"]=="Not Reported"].index)
set3=set3.drop(set3.loc[set3["light_condition_name"]=="Other"].index)
set3=set3.drop(set3.loc[set3["light_condition_name"]=="Unknown"].index)
# Combine "Dark – Lighted", "Dark – Not Lighted", and "Dark – Unknown Lighting" as one level:
set3.loc[set3.loc[set3["light_condition_name"]=="Dark – Lighted"].index,"light_condition_name"]="Dark"
set3.loc[set3.loc[set3["light_condition_name"]=="Dark – Not Lighted"].index,"light_condition_name"]="Dark"
set3.loc[set3.loc[set3["light_condition_name"]=="Dark – Unknown Lighting"].index,"light_condition_name"]="Dark"
# Combine "Blowing Sand, Soil, Dirt" and "Blowing Snow" as one level:
set3.loc[set3.loc[set3["atmospheric_conditions_1_name"]=="Blowing Snow"].index,"atmospheric_conditions_1_name"]="Blowing"
set3.loc[set3.loc[set3["atmospheric_conditions_1_name"]=="Blowing Sand, Soil, Dirt"].index,"atmospheric_conditions_1_name"]="Blowing"
# Change "Fog, Smog, Smoke" to "Fog" and "Freezing Rain or Drizzle" to "Freezing":
set3.loc[set3.loc[set3["atmospheric_conditions_1_name"]=="Fog, Smog, Smoke"].index,"atmospheric_conditions_1_name"]="Fog"
set3.loc[set3.loc[set3["atmospheric_conditions_1_name"]=="Freezing Rain or Drizzle"].index,"atmospheric_conditions_1_name"]="Freezing"
# Contengenccy table showing number of accidents:
weather_accident= set3.iloc[:, 0:3].pivot_table(index="light_condition_name", columns="atmospheric_conditions_1_name", values="accidents", aggfunc=np.sum)
weather_accident
# plot the table:
weather_accident.plot(kind="bar", figsize=(8,4), fontsize="large",width=1.6, legend="best")
plt.ylabel("Sum of accidents")
matplotlib.font_manager.FontProperties(variant="small-caps",stretch="ultra-condensed",size='xx-small')
# Change type of categorical columns:
set3["light_condition_name"]= set3["light_condition_name"].astype("category")
set3["atmospheric_conditions_1_name"]= set3["atmospheric_conditions_1_name"].astype("category")
# Mosaic plot: 
plt.figure(figsize=(20,20))
mosaic(data=weather_accident.stack())
plt.show()