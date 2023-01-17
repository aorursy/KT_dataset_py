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
earthquakes.Date.dtype
earthquakes.Date.head(15)
earthquakes.shape
earthquakes.info()
earthquakes.tail()
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
other_format = earthquakes.Date.str.findall('.*T.*Z')
wrong_dates = other_format[other_format.apply(len) > 0].apply(lambda x: x[0])
wrong_dates
converted_dates = pd.to_datetime(wrong_dates)
converted_dates
converted_dates[3378].day
converted_dates = converted_dates.apply(lambda x: '{}/{}/{}'.format(x.month, x.day, x.year))
converted_dates
earthquakes.Date.update(converted_dates)
earthquakes.Date.head()
other_format = earthquakes.Date.str.findall('.*T.*Z')
(other_format.apply(len) > 0).sum()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format='%m/%d/%Y')
print(earthquakes.date_parsed.dtype)
earthquakes.date_parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes.date_parsed.dt.day
day_of_month_earthquakes.head()
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(earthquakes.date_parsed.dt.day, bins=31)
volcanos['Last Known Eruption'].sample(5)
v_dates = volcanos['Last Known Eruption']
print(v_dates.shape)
n_total = v_dates.shape[0]
common_era = v_dates.str.findall('\d+ CE').apply(len) > 0
n_ce = common_era.sum()
n_ce
before_common_era = v_dates.str.findall('\d+ BCE').apply(len) > 0
n_bce = before_common_era.sum()
n_bce
print(n_total)
print(n_ce + n_bce)
dates_ce = v_dates[common_era]
dates_bce = v_dates[before_common_era]
dates_other = v_dates[~common_era]
dates_other = dates_other[~before_common_era]
print(dates_ce.head())
print(dates_bce.head())
dates_other.head()
(dates_other == 'Unknown').sum() / dates_other.shape[0]
dates_ce = dates_ce.apply(lambda x: pd.Period(year=int(x.split(' ')[0]), freq='Y'))
print(dates_ce.dtype)
dates_ce.head()
dates_ce.dt.year.head()  # Even if the dtype is "object" this is a Period object (could be used as PeriodIndex if required)
dates_bce = dates_bce.apply(lambda x: pd.Period(year=-int(x.split(' ')[0]), freq='Y'))
print(dates_bce.dtype)
dates_bce.head()
from pandas.tseries.offsets import YearEnd
dates_bce[5]
dates_bce[5] + YearEnd(105)
dates_other = dates_other.apply(lambda x: pd.NaT)
dates_other.head()
new_dates = pd.concat([dates_ce, dates_bce, dates_other])
print(new_dates.isnull().sum())
print(len(dates_other))
volcanos['Last Known Eruption'] = new_dates
volcanos.head()
new_dates.value_counts().head(10)
new_dates
new_dates.dt.year

sns.distplot(new_dates.dt.year)
pd.NaT.year
dates_other.head()
dates_other.dt.year.head()