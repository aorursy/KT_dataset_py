import seaborn as sns

import datetime

import numpy as np # linear algebra

import pandas as pd # data processing



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



np.random.seed(0)
# importing data

landslides = pd.read_csv("/kaggle/input/landslide-events/catalog.csv")

earthquakes = pd.read_csv("/kaggle/input/earthquake-database/database.csv")
# first look at landslides dataset

print(landslides.columns)

landslides.head()
# first look at earthquakes dataset

print(earthquakes.columns)

earthquakes.head()
# for landslides



# print the first few rows of the date column

print(landslides['date'].head())
# check the data type for date column

landslides['date'].dtype
# for earthquakes



# print the first few rows of the date column

print(earthquakes['Date'].head())
# for landslides 



# create a new column, date_parsed, with the parsed dates

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows

landslides['date_parsed'].head()
# for earthquake



# create a new column, date_parsed, with the parsed dates



earthquakes["date_parsed"] = pd.to_datetime(earthquakes["Date"], format="%m/%d/%Y", errors="coerce")



invalid_date_index = earthquakes["date_parsed"][earthquakes["date_parsed"].isnull() == True].index.tolist()
# print the first few rows

earthquakes['date_parsed'].head()
# for landslides



# try to get the day of the month from the date column

day_of_month_landslides = landslides['date'].dt.day
# for landslides



# get the day of the month from the date_parsed column 

day_of_month_landslides = landslides['date_parsed'].dt.day

day_of_month_landslides.head()
# for earthquakes



# get the day of the month from the date_parsed column 

day_of_month_earthquakes = earthquakes['date_parsed'].dt.day

day_of_month_earthquakes.head()
# for landslides



# remove na's

day_of_month_landslides = day_of_month_landslides.dropna()



# plot the day of the month

sns.distplot(day_of_month_landslides, kde = False, bins = 31)
# for earthquakes



#remove na's

day_of_month_earthquakes = day_of_month_earthquakes.dropna()



# plot the day of the month

sns.distplot(day_of_month_earthquakes, kde = False, bins = 31)