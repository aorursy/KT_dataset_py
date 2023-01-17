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
print(earthquakes['Date'].head())
earthquakes['Date'].dtype


# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")
# print the first few rows
landslides['date_parsed'].head()
# Your turn! Create a new column, date_parsed, in the earthquakes
# dzataset that has correctly parsed dates in it. (Don't forget to 
# double-check that the dtype is correct!)
#attempt at finding def that might do away with the unparsables
"""df2 = pd.DataFrame(earthquakes, columns=['Date'])

def myfunc(x):
    if len(x)==4:
        x = '01/01/'+x
    else:
        if not re.search('/',x):
            example = re.sub('[-]','/',x)
            terms = re.split('/',x)
            if (len(terms)==2):
                if len(terms[-1])==2:
                    x = '01/'+terms[0]+'/19'+terms[-1]
                else:
                    x = '01/'+terms[0]+'/'+terms[-1] 
            elif len(terms[-1])==2:
                x = terms[0].zfill(2)+'/'+terms[1].zfill(2)+'/19'+terms[-1]
    return x

df2['Date']""" #= df2.Date.str.replace(r'(((?:\d+[/-])?\d+[/-]\d+)|\d{4})', lambda x: myfunc(x.groups('Date')[0]))
#sort data to find those that did not parse
s=earthquakes.sort_values('Date') 
#print(s)

#remove rows that did not parse
"""s = s.drop(s.index(3378)) #[7512], [20650]))#['1975-02-23T02:58:41.000Z', '1985-04-28T02:53:41.530Z', '2011-03-13T02:23:34.520Z'])
s = s.drop(s.index(7512))
s = s.drop(s.index(20650))"""
s = s[:-3]
#print(s)
earthquakes['date_parsed'] = pd.to_datetime(s['Date'], format = "%m/%d/%Y")
earthquakes['date_parsed'].head()
#earthquakes['date_parsed'].dtype

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
# remove na's
day_of_month_earthquakes = day_of_month_earthquakes.dropna()

# plot the day of the month
sns.distplot(day_of_month_earthquakes, kde=False, bins=31)

volcanos['Last Known Eruption'].sample(5)