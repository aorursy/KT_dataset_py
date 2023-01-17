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
# now, check if the date rows are all dates.
print(earthquakes['Date'].head())
#Checking the data type of the Date column in the earthquakes dataframe
earthquakes['Date'].dtype

# Create a new column, date_parsed, in the earthquakes
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y") 
# dataset that has correctly parsed dates in it. (Don't forget to 
earthquakes('date_parsed').head()
# double-check that the dtype is correct!)
earthquakes['date_parsed'].dtype

print (pd.to_datetime(earthquakes['Date'], errors = 'coerce', format="%m/%d/%Y"))
mask = pd.to_datetime(earthquakes['Date'], errors = 'coerce', format="%m/%d/%Y").isnull()
print (earthquakes['Date'][mask])
earthquakes['date_Parsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)
earthquakes.date_Parsed.head()
# now i can go on like a normal human being and get the day of the month from the date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
# I see you small 'p'
# I have to try again, key error seems to be date_parsed
earthquakes['day_Month_earthquakes'] = earthquakes['date_Parsed'].dt.day
earthquakes['day_Month_earthquakes'].head()
# tremove na's
earthquakes.day_Month_earthquakes = earthquakes.day_Month_earthquakes.dropna()

# plot the day of the month
sns.distplot(earthquakes.day_Month_earthquakes, kde=False, bins=31)
volcanos['Last Known Eruption'].sample(5)