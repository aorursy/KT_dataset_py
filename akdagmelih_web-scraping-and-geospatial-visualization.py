from IPython.display import Image

Image("../input/hov-alvin/hov_alvin_1.jpg")
# Importing necessary libraries:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from bs4 import BeautifulSoup

import requests

import datetime

from warnings import filterwarnings

filterwarnings('ignore')
# Defining the url where the data is stored:

url_list = ['http://dsg.whoi.edu/divelog.nsf/By%20Pilot%20Name?OpenView&Start=1',

            'http://dsg.whoi.edu/divelog.nsf/By%20Pilot%20Name?OpenView&Start=5065&Count=6000']



# Create an empty list for storage:

data1 = []



# Using for loops to collects columns of data from the urls:

for each in url_list:

    r = requests.get(each)

    soup = BeautifulSoup(r.content, 'html.parser')

    

    for tr in soup.find_all('tr'):

        col = 0

        for td in tr.find_all('td'):

            td_text = td.get_text().strip()

            if col == 0:

                pilot = td_text

            if col == 1:

                dive_nu = td_text

            if col == 2:

                date = td_text

            if col == 3:

                op_area = td_text

            if col == 4:

                lat = td_text

            if col == 5:

                lon = td_text

            if col == 6:

                depth = td_text

            if col == 7:

                obs1 = td_text

            if col == 8:

                obs2 = td_text

            if col == 9:

                dive_time = td_text

            if col == 10:

                bottom_time = td_text if td_text else np.nan

        

                data1.append({'pilot': pilot,

                             'dive_nu': dive_nu,

                             'date': date,

                             'op_area': op_area,

                             'lat': lat,

                             'lon': lon,

                             'depth': depth,

                             'obs1': obs1,

                             'obs2': obs2,

                             'dive_time': dive_time,

                             'bottom_time': bottom_time})

            col += 1



# Turn collected data into Pandas Dataframe: 

df1 = pd.DataFrame(data1, columns=['pilot', 'dive_nu', 'date', 'op_area', 

                                   'lat', 'lon', 'depth', 'obs1', 'obs2', 'dive_time', 'bottom_time'])
# Erasing duplicated values:

df1.drop_duplicates(subset='dive_nu', keep='first' , inplace=True)

print('Is there any duplicated value? ', df1.duplicated().sum())



# Let's see what we have collected:

df1.head()
# Urls for the summary table:

url_list = ['http://dsg.whoi.edu/divelog.nsf/Summary?OpenView&Start=1', 

            'http://dsg.whoi.edu/divelog.nsf/Summary?OpenView&Start=5065']



# Create an empty list for storage:

data2 = []



# Collect data by for loops:

for each in url_list:

    r = requests.get(each)

    soup = BeautifulSoup(r.content, 'html.parser')

    

    for tr in soup.find_all('tr'):

        col = 0

        for td in tr.find_all('td'):

            td_text = td.get_text().strip()

            if col == 0:

                date = td_text

            if col == 1:

                dive_nu = td_text

            if col == 2:

                cruise = td_text

            if col == 3:

                leg = td_text

            if col == 4:

                chief_sci = td_text

                

                data2.append({'date': date,

                              'dive_nu': dive_nu,

                              'cruise': cruise,

                              'leg': leg,

                              'chief_sci': chief_sci})

            col += 1



# Convert the data into Pandas Dataframe:

df2 = pd.DataFrame(data2, columns=['date', 'dive_nu', 'cruise', 'leg', 'chief_sci'])



# Erase the first row from the dataset which is the website menu items: 

df2 = df2[1:5066]



# Let's see what we have collected:

df2.head()
# Urls for the summary table:

url_list = ['http://dsg.whoi.edu/divelog.nsf/By%20Dive%20Number/Date?OpenView',

            'http://dsg.whoi.edu/divelog.nsf/By%20Dive%20Number/Date?OpenView&Start=5065']



data3 = []



# Collecting data from the web site:

for each in url_list:

    r = requests.get(each)

    soup = BeautifulSoup(r.content, 'html.parser')

    

    for tr in soup.find_all('tr'):

        col = 0

        for td in tr.find_all('td'):

            td_text = td.get_text().strip()

            if col == 0:

                dive_nu = td_text

            if col == 1:

                date = td_text

            if col == 2:

                purpose = td_text

                

                data3.append({'dive_nu': dive_nu, 'date': date, 'purpose': purpose})

            col += 1



# Converting data into Pandas Dataframe:            

df3 = pd.DataFrame(data3, columns=['dive_nu', 'date', 'purpose'])



# Erase website menu items from the dataset 

df3 = df3[1:5066]
# Let's see our three datasets:

display(df1.head(2))

display(df2.head(2))

display(df3.head(2))
# Merging the datasets:

data_raw = pd.merge(df1, df2, on=['date', 'dive_nu'], how='outer')

data_raw = pd.merge(data_raw, df3, on=['date', 'dive_nu'], how='outer')



# Changing the order of the columns:

data_raw = data_raw[['dive_nu', 'date', 'op_area', 'lat', 'lon', 'cruise', 

                     'leg', 'purpose', 'depth', 'dive_time', 'bottom_time',

                     'chief_sci', 'pilot', 'obs1', 'obs2']]



display(data_raw.head(3))
# data_raw.to_csv(r'/Users/melihakdag/Desktop/Data Science/alvin_dive_logs/alvin_data_raw.csv')
data_raw.info()
# Changing 'date' column type into datetime:

data_raw['date'] = pd.to_datetime(data_raw['date'])

data_raw.tail(3)
# Define the function:

def change_year(x):

    if x.year > 2060:

        year = x.year - 100

        

    else:

        year = x.year



    return datetime.date(year,x.month,x.day)



# Apply the function:

data_raw['date'] = data_raw['date'].apply(change_year)



# Let's see if the dates are correct now:

data_raw.tail(3)
# Changing 'dive_time' and 'bottom_time' column types into minutes:

def minutes(x):

    if type(x) == str:

        return int(x[:-3])*60 + int(x[-2:])

    else:

        return np.nan

        

data_raw['bottom_time'] = data_raw['bottom_time'].apply(minutes)

data_raw['dive_time'] = data_raw['dive_time'].apply(minutes)
# Changing 'dive_nu', 'depth' columns' types into integer:

to_integer = lambda x: int(x)



columns = ['dive_nu', 'depth']



for each in columns:

    data_raw[each] = data_raw[each].apply(to_integer)
# Splitting latitude and longitude strings by '-':

data_raw[['lat01', 'lat02']] = data_raw['lat'].str.split(pat="-", expand=True)

del data_raw['lat']



data_raw[['long01', 'long02']] = data_raw['lon'].str.split(pat='-', expand=True)

del data_raw['lon']
data_raw.head(2)
# Let's see if there is any missing values in the new features:

data_raw[['lat01', 'lat02', 'long01', 'long02']].isnull().values.any()
# Which columns have the missing values? 

print('lat01 any NaN : ', data_raw['lat01'].isnull().values.any())

print('lat02 any NaN : ', data_raw['lat02'].isnull().values.any())

print('long01 any NaN: ', data_raw['long01'].isnull().values.any())

print('long02 any NaN: ', data_raw['long02'].isnull().values.any())
# Find those missing values in long02 column:

data_raw[data_raw['long02'].isnull() == True]
# It looks like the missing values are caused by a typo. Let's fix the typos: 

data_raw.iloc[3508, 15] = data_raw.iloc[3508, 15].replace('95.28.5W', '95')

data_raw.iloc[3508, 16] = '28.5W'



data_raw.iloc[3514, 15] = data_raw.iloc[3514, 15].replace('95.33.0W', '95')

data_raw.iloc[3514, 16] = '33.0W'



data_raw.iloc[[3508, 3514]]
# Changing latidude and longitude strings into numbers:

data_raw['lat01'] = data_raw['lat01'].apply(lambda x: float(x))

data_raw['long01'] = data_raw['long01'].apply(lambda x: float(x))
# Changing latitude and longitude degree signs according to being on the Southern hemisphere or having a Western longitude:

def change_sign(x, y):

    if y[-1] == 'S' or y[-1] == 'W':

        return x*-1

    else:

        return x

    

data_raw['lat01'] = data_raw.apply(lambda x: change_sign(x.lat01, x.lat02), axis=1)

data_raw['long01'] = data_raw.apply(lambda x: change_sign(x.long01, x.long02), axis=1)



data_raw.head(3)
# Changing Decimal Minutes to Decimal Degrees:

def to_DD(x):

    if x[-1:] == 'N':

        return float(x[:-1])/60

    if x[-1:] == 'S':

        return float(x[:-1])/-60

    if x[-1:] == 'E':

        return float(x[:-1])/60

    if x[-1:] == 'W':

        return float(x[:-1])/-60



# Finding Decimal Degrees:

data_raw['lat02'] = data_raw['lat02'].apply(to_DD)

data_raw['long02'] = data_raw['long02'].apply(to_DD)



# Saving the DD latidude and longitudes and deleting other coordinate datas:

data_raw['lat(DD)'] = data_raw['lat01'] + data_raw['lat02']

data_raw['long(DD)'] = data_raw['long01'] + data_raw['long02']

data_raw.drop(['lat01', 'lat02', 'long01', 'long02'], axis=1, inplace=True)
# Data inputs are written in different styles:

data_raw[['op_area','purpose', 'chief_sci']].sample(5)
# Let's make the string styles all the same:

data_raw[['op_area','purpose', 'chief_sci']] = data_raw[['op_area','purpose', 'chief_sci']].apply(lambda x: x.str.title())
# Create a copy of the final data set:

alvin_dives = data_raw.copy()



# Finally we can export this cleaned dataset:

#alvin_dives.to_csv(r'/Users/melihakdag/Desktop/Data Science/alvin_dive_logs/alvin_data_cleaned.csv')



alvin_dives.head()
import plotly.express as px

import folium

from folium import Circle
# Defining the borders of our map:

sw = alvin_dives[['lat(DD)', 'long(DD)']].min().values.tolist()

ne = alvin_dives[['lat(DD)', 'long(DD)']].max().values.tolist()



# Creating the basemap:

world_map = folium.Map(zoom_start = 9)

# Default zoom setting with the boundaries:

world_map.fit_bounds([sw, ne])



# Creating circle marks on the basemap:

for lat, long, date, depth in zip(alvin_dives['lat(DD)'], alvin_dives['long(DD)'], 

                                       alvin_dives['date'], alvin_dives['depth']): 

    folium.Circle(location = [lat, long], 

                  radius = 20).add_child(folium.Popup(str(date.year) + ', ' + str(depth) + ' m')).add_to(world_map)

    

world_map
# We can see total number of dives for each year.

# Total dives for the first ten years:

dive_years = pd.DataFrame(alvin_dives.groupby(alvin_dives['date'].map(lambda x: x.year)).dive_nu.count())

dive_years.reset_index(inplace=True)

dive_years.rename(columns={'date':'year', 'dive_nu':'total_dive_nu'}, inplace=True)

dive_years.head()
fig = px.line(dive_years, 

              x="year", 

              y="total_dive_nu", 

              title='Total Number of Dives in Years')



fig.show()
# How many different purposes are there?

print('Alvin have dived for %d different purposes.' %len(alvin_dives['purpose'].unique()))
# Let's see the distribution of the purposes:

purpose_count = pd.DataFrame(alvin_dives['purpose'].value_counts())

purpose_count.reset_index(inplace=True)

purpose_count.rename(columns = {'index':'purpose', 'purpose':'total_nu'}, inplace=True)

purpose_count.head(10)
fig = px.bar(purpose_count.head(20),

             x='purpose',

             y='total_nu',

             title='Top 20 Purposes of the Dives')



fig.update_traces(marker=dict(color="RoyalBlue"))



fig.show()
dive_depths = alvin_dives[['date', 'dive_nu', 'depth', 'dive_time']]

dive_depths['depth'] = dive_depths['depth'].map(lambda x: -(x))
fig = px.scatter(dive_depths,

                 x = 'date', 

                 y = 'depth',

                 title = "Alvin's Dive Dates & Depths & Times",

                 color='dive_time',

                 hover_data = ['dive_nu'])

fig.show()
# Group by purpose and find the average depth for each purpose:

purpose_avg_depth = pd.DataFrame(alvin_dives.groupby(['purpose']).depth.median())

purpose_avg_depth.reset_index(inplace=True)

purpose_avg_depth.rename(columns={'depth':'avg_depth'}, inplace=True)
# Combine the purpose count and the average depth tables:

purpose_depth_count = pd.merge(purpose_count, purpose_avg_depth, on=['purpose'])

purpose_depth_count['avg_depth'] = purpose_depth_count['avg_depth'].map(lambda x: -(x))

purpose_depth_count.head(10)
fig = px.scatter(purpose_depth_count.head(20), 

                 x = 'purpose',

                 y = 'avg_depth',

                 size = 'total_nu',

                 title = 'Average Depths for Purposes')   

fig.show()
max_dive_time = pd.DataFrame(alvin_dives.groupby(['depth'])['dive_time'].max())

max_dive_time.reset_index(inplace=True)

max_dive_time.rename(columns={'dive_time': 'max_dive_time'}, inplace=True)

max_dive_time['depth'] = max_dive_time['depth'].map(lambda x: -(x)) 

max_dive_time.tail()
fig = px.scatter(max_dive_time, 

                 x = 'max_dive_time',

                 y = 'depth',

                 color = 'depth',

                 title = 'Maximum Dive Times for Depths')

fig.show()
# Let's extract the pilots, chief scientists, observers,

divers = alvin_dives[['pilot', 'chief_sci', 'obs1', 'obs2', 'date', 'dive_nu', 'purpose']]

divers['date'] = divers['date'].map(lambda x: x.year)
print('%d talented people had chance to work as a pilot for ALVIN since 1964.' %len(divers['pilot'].unique()))
divers_dives = pd.DataFrame(divers.groupby(['pilot', 'date']).dive_nu.count())

divers_dives.reset_index(inplace=True)

divers_dives.rename(columns={'date': 'year', 'dive_nu': 'total_dives'}, inplace=True)

divers_dives.sort_values(by='year', inplace=True)

divers_dives.head()
fig = px.bar(divers_dives, 

             y = 'pilot',

             x = 'total_dives',

             color = 'year',

             title = 'Pilots and Total Dive Numbers')

fig.show()
fig = px.scatter(divers_dives, 

                 y = 'pilot',

                 x = 'year',

                 color = 'year',

                 title = 'Pilots and Years')

fig.show()
chief_scientist = pd.DataFrame(divers.groupby(['chief_sci', 'date']).dive_nu.count())

chief_scientist.reset_index(inplace=True)

chief_scientist.rename(columns={'date': 'year', 'dive_nu': 'total_dives'}, inplace=True)

chief_scientist.sort_values(by='year', ascending=True, inplace=True)
chief_scientist.head()
fig = px.scatter(chief_scientist,

                 x = 'year',

                 y = 'chief_sci',

                 color = 'year',

                 size = 'total_dives',

                 title = 'Chief Scientists and Research Years')

fig.show()
# Cruises and total number of dives:

cruise_dives = pd.DataFrame(alvin_dives.groupby('cruise').dive_nu.count().sort_values(ascending=False))

cruise_dives.rename(columns={'dive_nu':'total_dives'}, inplace=True)



# Getting the cruises which have more than 50 dives:

cruise_dives50 = cruise_dives[cruise_dives.values > 50]
fig = px.pie(cruise_dives50,

             names = cruise_dives50.index,

             values = 'total_dives',

             title = 'Cruises With More Than 50 Dives')

fig.show()