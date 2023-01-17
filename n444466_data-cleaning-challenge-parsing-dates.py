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
display(earthquakes.head())

# earthquakes.Date.dtype
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
display(landslides.head())
landslides['date_parsed'].dtype
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
print(earthquakes["Date"].head())

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")  # year is 4 digitals so it should be Y

#@@@ Cheat and Fast way to solve the format problem. # learn from https://www.kaggle.com/giodev11
#earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)


## some problems occur, let's do some inspection
##Solution1: learn from  https://www.kaggle.com/giodev11
earthquakes['lenDate'] = earthquakes['Date'].apply(len)
earthquakes.loc[earthquakes['lenDate'] > 10]


## Solution2: Find where the date is poorly formated   Note: Learn from https://www.kaggle.com/pmmaraujo
## earthquakes.Date[earthquakes['Date'].str.contains('-')] # its poorly formated in 3 fields only
earthquakes.loc[3378, 'Date']= '02/23/1975' 
earthquakes.loc[7512, 'Date']= '04/28/1985'
earthquakes.loc[20650, 'Date']= '03/13/2011' 
display(earthquakes.loc[[3378, 7512, 20650]])
#Then we can run the datetime format
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y") 
display(earthquakes.head())
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
print(day_of_month_landslides.head())
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes["date_parsed"].dt.day
print(day_of_month_earthquakes.head())
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.

# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

volcanos['Last Known Eruption'].sample(5)