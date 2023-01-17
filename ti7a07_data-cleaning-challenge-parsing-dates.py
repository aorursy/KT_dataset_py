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
earthquakes.head(2)
landslides.head(2)
volcanos.head(2)
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.columns
earthquakes.Date.dtype
earthquakes.Time.dtype
landslides.date.head(2)
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
landslides.date_parsed.head(2)
landslides.date_parsed.dtype
# print the first few rows
landslides['date_parsed'].head()
earthquakes.Date.head(150)
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsing'] = pd.to_datetime(earthquakes.Date, infer_datetime_format=True)
earthquakes.date_parsing.head(2)
earthquakes.date_parsing.dtype
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column
day_of_the_month_earthquakes = earthquakes.date_parsing.dt.day
day_of_the_month_earthquakes.head(10)
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
num_of_nulls = day_of_the_month_earthquakes.isnull().sum()
sns.distplot(day_of_the_month_earthquakes)
volcanos['Last Known Eruption'].sample(5)
volcanos['Last Known Eruption'].mode()
volcanos['Last Known Eruption'].dtype
copy_volcanos = volcanos
ind = copy_volcanos.index[x['Last Known Eruption'] == 'Unknown'].tolist()
copy_volcanos = copy_volcanos.drop(copy_volcanos.index[ind])

copy_volcanos['Last Known Eruption'].sample(5)
BCE = copy_volcanos['Last Known Eruption'].str.contains(' BCE')
CE = copy_volcanos['Last Known Eruption'].str.contains(' CE')

BCE_ = copy_volcanos['Last Known Eruption'][BCE].str.replace(' BCE','')
CE_ = copy_volcanos['Last Known Eruption'][CE].str.replace(' CE','')
CE_ = CE_.astype(int)
BCE_ = - BCE_.astype(int)
apped = CE_.append(BCE_)
copy_volcanos['Last Known Eruption'] = apped
copy_volcanos['Last Known Eruption'].sample(10)

