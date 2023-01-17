import pandas as pd
import numpy as np
import seaborn as sns
import datetime

earthquakes = pd.read_csv("../input/earthquake-database/database.csv")
landslides = pd.read_csv("../input/landslide-events/catalog.csv")
volcanos = pd.read_csv("../input/volcanic-eruptions/database.csv")

# set seed for reproducibility
np.random.seed(0)
# check the data type of the Date column in the earthquakes dataframe
#check numpy documentation to match letter code to dypte of the object
#'O' is the code for object

earthquakes['Date'].dtype
landslides['date'].head(1)
# parsing dates: take in a string and identify its component parts
#tell pandas what the format of our dates are with a . 'strftime directive'
#point out which parts of the date are where, and what punctuation is between them
#example: 17-1-2007 has format"%d-%m-%Y" (capital Y is four digit year)
#create a new column, date_parsed, with the parsed dates

landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
#parse dates from earthquake
#have to use infer datetime format because there are multiple date formats in the column
#don't always use it because it's not always correct, and it's very slow

earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format=True)
earthquakes['date_parsed'].head(1)

# now that our date are in proper datetime format, we can use datetime to call specific values from it
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# can make sure we didn't mix up days and months by plotting a histogram; expect values to be b/w 1 and 31
# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)