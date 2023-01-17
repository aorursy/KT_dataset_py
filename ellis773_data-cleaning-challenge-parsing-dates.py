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

earthquakes['Date']

# create a new column, date_parsed, with the parsed dates

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows

landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes

# dataset that has correctly parsed dates in it. (Don't forget to 

# double-check that the dtype is correct!)



# Parsing the dates

# Using 'infer_datetime_format because there are dates that have differ format.

earthquakes['parsed_date'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)

earthquakes['parsed_date'].head()

# try to get the day of the month from the date column

day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column

day_of_month_landslides = landslides['date_parsed'].dt.day

day_of_month_landslides.head()
# Your turn! get the day of the month from the date_parsed column

day_of_month_earthquake = earthquakes['parsed_date'].dt.day

day_of_month_earthquake.head()
# remove na's

day_of_month_landslides = day_of_month_landslides.dropna()



# plot the day of the month

sns.distplot(day_of_month_landslides, kde=False, bins=31)
# Your turn! Plot the days of the month from your

# earthquake dataset and make sure they make sense.



from mlxtend.preprocessing import minmax_scaling



# Remove NA

day_of_month_earthquake = day_of_month_earthquake.dropna()



# Graph

scaled_data = minmax_scaling()

sns.distplot(day_of_month_earthquake, kde = False, bins = 31)

# Seperate CE and BCE



BCE_row = volcanos['Last Known Eruption'].str.contains(' BCE')

BCE = volcanos['Last Known Eruption'][BCE_row].str.replace(' BCE','')

CE_row = volcanos['Last Known Eruption'].str.contains(' CE')

CE = volcanos['Last Known Eruption'][CE_row].str.replace(' CE','')



# Convert Datatype

BCE = -BCE.astype(int)

CE = CE.astype(int)



# Append the two data

last_eruption = BCE.append(CE)



# Plot graph

sns.distplot(last_eruption)