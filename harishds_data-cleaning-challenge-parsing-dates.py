# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.sample();
earthquakes.Date.head()
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'],format="%m/%d/%Y",infer_datetime_format=True)
earthquakes.date_parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes.date_parsed.dt.day
day_of_month_earthquakes.sample()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes = day_of_month_earthquakes.dropna()
sns.distplot(day_of_month_earthquakes,kde=False,bins=31)
volcanos['Last Known Eruption'].sample(5)
# checking whether any null values are exist 
volcanos['Last Known Eruption'].isnull().sum()
# We are extracting the data from volcanos where 'Last Known Eruption' column is not 'Unknown'
volcanos_not_unknown = volcanos.loc[volcanos['Last Known Eruption'].str.extract('(Unknown)',expand=False).isnull()]
volcanos_not_unknown.sample()
#checking counts of volcanos, volcanos_not_unknown, volcanos_unknown
volcanos_unknown_len=volcanos['Last Known Eruption'].str.contains('Unknown').sum()
print("Length of volcanos : ",len(volcanos), 'Length of volcanos_not_unknown : ',len(volcanos_not_unknown), 'Length of volcanos_unknown : ',volcanos_unknown_len)
# 1508-871-637 = 0, its good counts are matched
volcanos_not_unknown['Eruption_date']=volcanos_not_unknown['Last Known Eruption'].apply(lambda date: -int(date.split(' ')[0]) if date.endswith('BCE')                                                                                else int(date.split(' ')[0]))
# getting sample values for date columns
date_of_volcanos_not_unknown=volcanos_not_unknown['Eruption_date']
date_of_volcanos_not_unknown.head()
#Finally plot the date
sns.distplot(date_of_volcanos_not_unknown,kde=False)