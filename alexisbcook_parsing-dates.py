# modules we'll use

import pandas as pd

import numpy as np

import seaborn as sns

import datetime



# read in our data

landslides = pd.read_csv("../input/landslide-events/catalog.csv")



# set seed for reproducibility

np.random.seed(0)
landslides.head()
# print the first few rows of the date column

print(landslides['date'].head())
# check the data type of our date column

landslides['date'].dtype
# create a new column, date_parsed, with the parsed dates

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
# print the first few rows

landslides['date_parsed'].head()
# get the day of the month from the date_parsed column

day_of_month_landslides = landslides['date_parsed'].dt.day

day_of_month_landslides.head()
# remove na's

day_of_month_landslides = day_of_month_landslides.dropna()



# plot the day of the month

sns.distplot(day_of_month_landslides, kde=False, bins=31)