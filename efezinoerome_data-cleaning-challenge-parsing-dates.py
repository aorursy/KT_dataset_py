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
earthquakes.Date.head()
# Your turn! Check the data type of the Date column in the earthquakes dataframe
# (note the capital 'D' in date!)
earthquakes.Date.dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
earthquakes['Date'].head(20)
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
earthquakes['Date_Parsed'] = pd.to_datetime(earthquakes['Date'],infer_datetime_format = True)
#pd.to_datetime(earthquakes['Date'][0:3379],format = "%m/%d/%Y")
print(earthquakes.Date_Parsed.dtype)
earthquakes.Date_Parsed.head()
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
# Your turn! get the day of the month from the date_parsed column
earthquakes.Date_Parsed.dt.day[0:5]
day_month = earthquakes.Date_Parsed.dt.day
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
print(len(day_month))
print(day_month.isnull().sum())
print(len(day_month) - day_month.isnull().sum())
print(day_month.isnull().sum(axis = 0))
#contains no missing values
# Your turn! Plot the days of the month from your
# earthquake dataset and make sure they make sense.
sns.distplot(day_month, kde = False)
earthquakes[['Date',"Date_Parsed"]].head()
volcanos['Last Known Eruption'].sample(5)
Unknown_val = volcanos['Last Known Eruption'].astype(str).apply(lambda x: x if len(x.split(" "))==0 else \
       x.split(" ")[len(x.split(" "))-1]) == "Unknown"
Period = volcanos['Last Known Eruption'].astype(str).apply(lambda x: x if len(x.split(" "))==0 else \
       x.split(" ")[len(x.split(" "))-1]) 
set(volcanos['Last Known Eruption'].astype(str).apply(lambda x: x if len(x.split(" "))==0 else \
       x.split(" ")[len(x.split(" "))-1]))
volcanos_last_known_eruption_no_unknown = volcanos['Last Known Eruption'][~(Unknown_val)]
volcanos_last_known_eruption_no_unknown.apply(lambda x: 1 - int(x.split(" ")[0]) if x.split(" ")[1] == "BCE" else \
             int(x.split(" ")[0])).values.tolist()
pd.concat([volcanos_last_known_eruption_no_unknown.apply(lambda x: 1 - int(x.split(" ")[0]) if x.split(" ")[1] == "BCE" else \
             int(x.split(" ")[0])),volcanos_last_known_eruption_no_unknown], axis = 1)