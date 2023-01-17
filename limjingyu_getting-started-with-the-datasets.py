import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno # check for missing values



# Input data files are available in the "../input/" directory.

# reserves

air_reserve = pd.read_csv("../input/air_reserve.csv")

hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")



# others

visits = pd.read_csv("../input/air_visit_data.csv")

dates = pd.read_csv("../input/date_info.csv")

relation = pd.read_csv("../input/store_id_relation.csv")



# store info

air_store_info = pd.read_csv("../input/air_store_info.csv")

hpg_store_info = pd.read_csv("../input/hpg_store_info.csv")

# visualization of missing data, white fields indicate NAs

# not plotting for relation and date info since it's meant to provide basic information

msno.matrix(air_reserve)

msno.matrix(hpg_reserve)

msno.matrix(visits)

msno.matrix(air_store_info)

msno.matrix(hpg_store_info)
air_reserve.head()
hpg_reserve.head()
# combine reserve datasets since store_id for air and hpg have labels

# renaming columns

cols = ["store_id", "visit_datetime", "reserve_datetime", "reserve_visitors"]

air_reserve.columns = cols 

hpg_reserve.columns = cols



# creating a new dataframe with new column names

reserves = pd.DataFrame(columns=cols)

reserves = pd.concat([air_reserve, hpg_reserve])



reserves.info()

reserves.describe()
print("Number of restaurants with reservations from AirREGI = ", str(len(air_reserve['store_id'].unique())))

print("Number of restaurants with reservations from hpg = ", str(len(hpg_reserve['store_id'].unique())))
# plot number of visitors per reservation

sns.set(color_codes=True)

visitors = reserves["reserve_visitors"]

sns.distplot(visitors)
# plot number of visits to each air restaurant

sns.set(color_codes=True)

visitors = visits["visitors"]

sns.distplot(visitors, color="y")
visits.info()

visits.describe()
print("Number of Air restaurants = ", str(len(visits["air_store_id"].unique())))
relation.info()
dates.head()
print("Number of Air restaurants = ", str(len(air_store_info)))

print("Number of hpg restaurants = ", str(len(hpg_store_info)))
air_store_info.info()

air_store_info.head()
print("Genres:")

air_store_info["air_genre_name"].unique()
print("Number of unique areas = ", str(len(air_store_info["air_area_name"].unique())))
# unique areas (expand for the list)

air_store_info["air_area_name"].unique()
hpg_store_info.info()

hpg_store_info.head()
print("Genres:")

hpg_store_info["hpg_genre_name"].unique()
print("Number of unique areas = ", str(len(hpg_store_info["hpg_area_name"].unique())))
# unique areas (expand for the list)

hpg_store_info["hpg_area_name"].unique()