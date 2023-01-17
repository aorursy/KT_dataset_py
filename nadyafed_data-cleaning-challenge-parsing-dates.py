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

earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows

landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes

# dataset that has correctly parsed dates in it. (Don't forget to 

# double-check that the dtype is correct!)

#indices = np.where([earthquakes['Date'].str.len().value_counts()==24])[1]

#earthquakes.loc[indices]

earthquakes['date_parsed']=pd.to_datetime(earthquakes['Date'], format = "%m/%d/%y", infer_datetime_format = True)

print(earthquakes['date_parsed'].dtype)
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

earthquakes['date_parsed'].isna().value_counts() # check for NaN values

sns.distplot(day_of_month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(25)
volcanos_eruption_years_df = pd.DataFrame({ "BCE": volcanos.loc[:,'Last Known Eruption'].str.find(' BCE').value_counts(),

                                           "CE": volcanos.loc[:,'Last Known Eruption'].str.find(' CE').value_counts(),

                                           "Unknown": volcanos.loc[:,'Last Known Eruption'].str.find('Unknown').value_counts()

                                          })

volcanos_eruption_years_df.head(10)
volcanos_unknown_indices = volcanos.loc[:,'Last Known Eruption'].str.find('Unknown') >-1

volcanos.loc[volcanos_unknown_indices, 'eruption_date_parsed'] = np.NaN
volcanos_bce = volcanos.loc[:,'Last Known Eruption'].str.find(' BCE') >-1

volcanos_bce_indices = volcanos.loc[volcanos_bce, 'Last Known Eruption'].index

s = volcanos.loc[volcanos_bce_indices,'Last Known Eruption'].str.rpartition(' ')[[0]].astype(str)

volcanos.loc[volcanos_bce_indices,'eruption_date_parsed'] = '-'+s[0]

volcanos.head(20)
volcanos_ce = volcanos.loc[:,'Last Known Eruption'].str.find(' CE') >-1

volcanos_ce_indices = volcanos.loc[volcanos_ce, 'Last Known Eruption'].index

s = volcanos.loc[volcanos_ce_indices,'Last Known Eruption'].str.rpartition(' ')[[0]].astype(str)

volcanos.loc[volcanos_ce_indices,'eruption_date_parsed'] = s[0]

volcanos.head(20)
volcanos['eruption_date_parsed'].isna().sum() # = 637, same as in the original column, meaning all the values have been parsed
volcanos_eruption_date = volcanos['eruption_date_parsed'].dropna()

sns.distplot(volcanos_eruption_date.astype(int), kde=False, bins=12)