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
earthquakes['Date'].head()

# Checking the data type of the Date column in earthquakes dataframe
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format="%m/%d/%Y", infer_datetime_format=True)
earthquakes['date_parsed'].dtype

earthquakes['date_parsed'].head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
# Removing na's from the day of the month - earthquakes 
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# Plotting the distribution of the data of the month - earthquakes
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

# Getting 5 sample observations from the Last Known Eruption Column from Volcanos Dataframe
volcanos['Last Known Eruption'].sample(5)
# Getting a boolean column for the observations that ends with 'CE' in Last Known Eruption Column
volcanos['Last_Known_Eruption_CE'] = volcanos['Last Known Eruption'].str.endswith('CE')

# Checking a sample of 5 observations
volcanos['Last_Known_Eruption_CE'].sample(5)
# Passing the boolean column as a condition to select 
volcanos['parsed_dates'] = volcanos['Last Known Eruption'][volcanos['Last_Known_Eruption_CE']]

# Checking sample of 5 observations
volcanos['parsed_dates'].sample(5)
# As we saw there are some NaN values. Removing the NaN values using dropna
last_known_eruption_dates = volcanos['parsed_dates'].dropna()

# Checking sample of 5 observations
last_known_eruption_dates.sample(5)
# Getting the years alone from the last_known_eruption_dates
last_known_eruption_dates = last_known_eruption_dates.apply(lambda x: -int(x[:-4]) if x.endswith("BCE") else int(x[:-3]))
# .apply applies the lambda function to all the observations of last_known_eruption-dates
# Lambda gets all letters of the string except last 4 letters for the values that ends with 'BCE' 
# else it gets all letters of the string except last 3 letters

# Checking the first 5 observations
last_known_eruption_dates.head()
# Plotting the years as a distribution plot
sns.distplot(last_known_eruption_dates, kde=False)