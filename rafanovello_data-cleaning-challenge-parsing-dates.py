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
earthquakes.info()
landslides.info()
volcanos.info()
# print the first few rows of the date column
print(landslides['date'].head())
# check the data type of our date column
landslides['date'].dtype
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes.Date)
earthquakes['date_parsed'].dtype
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31);
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
day_of_month_earthquakes.dropna(inplace=True)
sns.distplot(day_of_month_earthquakes, kde=False, bins=31);
volcanos['Last Known Eruption'].sample(5)
from dateutil.parser import parse

def parse_eruptions(cel):
    cel = cel.lower()
    if 'unknown' in cel:
        return np.nan
    if 'bce' in cel or 'ce' in cel:
        return parse(cel, fuzzy_with_tokens=True)

volcanos['eruption_parsed'] = volcanos['Last Known Eruption'].apply(parse_eruptions)
volcanos['eruption_parsed'].sample(5)
pd.to_datetime(volcanos['eruption_parsed'], origin='julian', unit='D')
import dateutil
dateutil.parser.parse('2000 CE', fuzzy_with_tokens=True)
