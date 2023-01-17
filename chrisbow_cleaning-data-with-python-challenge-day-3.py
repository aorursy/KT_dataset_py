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
# take a look at the variable names
print(earthquakes.head())
# note the capital 'D' indeed, let's answer the question...
earthquakes['Date'].dtype
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")
print (pd.to_datetime(earthquakes['Date'], errors = 'coerce', format="%m/%d/%Y"))
mask = pd.to_datetime(earthquakes['Date'], errors = 'coerce', format="%m/%d/%Y").isnull()
print (earthquakes['Date'][mask])
earthquakes['dateParsed'] = pd.to_datetime(earthquakes['Date'], infer_datetime_format = True)
earthquakes.dateParsed.head()
earthquakes['dayOfMonth'] = earthquakes['dateParsed'].dt.day
earthquakes.dayOfMonth.head()
# remove na's
earthquakes.dayOfMonth = earthquakes.dayOfMonth.dropna()

# plot the day of the month
sns.distplot(earthquakes.dayOfMonth, kde = False, bins = 31)