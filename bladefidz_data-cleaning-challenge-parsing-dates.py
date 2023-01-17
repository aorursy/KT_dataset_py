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

# Inspect
# earthquakes.head()

earthquakes['Date'].dtype
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)

# Inspect format of Date column
formats = []
uniqueDateFormat = []
formatRows = []
for i, ud in zip(range(len(earthquakes['Date'])), earthquakes['Date']):
    f = [-1, -1]
    f[0] = ud.find('/')
    if f[0] is not -1:
        f[1] = ud[f[0]+1].find('/')
    if f not in formats:
        formats.append(f)
        uniqueDateFormat.append(ud)
        formatRows.append([i])
    else:
        formatRows[formats.index(f)].append(i)
uniqueDateFormat  # Malformat in index 1
# formatRows[1]

# earthquakes['Date'].head()

# Inspect valid format
print('Top 10 valids')
for i in formatRows[0]:
    print(earthquakes['Date'][i]) # Yup, %m/%d/%y
    if i > 10:
        break

# Subtitue malformed date in sane format 
# import re
print('\nFixing')
for i in formatRows[1]:
#     earthquakes['Date'][i] = earthquakes['Date'][i][:len(formatRows[0])-1]
#     earthquakes['Date'][i] = re.sub(r'-', r'/', earthquakes['Date'][i])
    dateToChange = earthquakes.loc[i, 'Date']
    print(dateToChange)
    goodDate = []
    goodDate += dateToChange[5:7].split()  # mounth
    goodDate.append('/')
    goodDate += dateToChange[8:10].split()  # day
    goodDate.append('/')
    goodDate += dateToChange[0:4].split()  # year
#     print(''.join(goodDate))
    earthquakes.loc[i, 'Date'] = ''.join(goodDate)
#     print(earthquakes['Date'][i])

# Inspect parsed format
print('\nFixed')
for i in formatRows[1]:
    print(earthquakes.loc[i, 'Date'])

# # Transform
earthquakes['date_parsed'] = pd.to_datetime(earthquakes['Date'], format = "%m/%d/%Y")
print('\nFixed type')
print(earthquakes['date_parsed'].dtype)
# try to get the day of the month from the date column
day_of_month_landslides = landslides['date'].dt.day
# get the day of the month from the landslides date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides
# Your turn! get the day of the month from the earthquake date_parsed column
day_of_month_earthquakes = earthquakes['date_parsed'].dt.day
day_of_month_earthquakes
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