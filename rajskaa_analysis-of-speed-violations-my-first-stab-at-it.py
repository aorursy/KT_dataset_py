# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



#other

from datetime import datetime
df = pd.read_csv('../input/cameras.csv')
df.shape
df.columns.values
df.dtypes
df.head()
df.tail()
df[5:15]
df.describe(include='all')
df.info()
df.replace('NaN', -1, inplace=True)

(df['LATITUDE'] == -1).count()
df[df['LATITUDE'] == -1]
df.dtypes
df.shape
df.columns
df.drop('LOCATION', axis=1, inplace=True)
df.info()
df[5:15]
df[df['ADDRESS'] == '2912 W ROOSEVELT']
df['ADDRESS'].unique()
def find_one(substrs, superstr):

    for substr in substrs:

        if superstr.find(substr) != -1:

            return substr

    return ''



address_values = df['ADDRESS'].values

street_values = []

for name_value in address_values:

    street_values.append(find_one(['N PULASKI', 'W PERSHING', 'S WESTERN', 'W JACKSON', 

                                   'W MADISON', 'N HUMBOLDT', 'W AUGUSTA', 'S COTTAGE GROVE', 

                                   'S KEDZIE', 'W 71ST', 'W ROOSEVELT', 'W OGDEN', 

                                   'W PETERSON', 'W FOSTER','N WESTERN', 'W ADDISON', 

                                   'E MORGAN DR', 'S PULASKI','S ARCHER', 'S STATE', 'E 95TH', 

                                   'W FULLERTON', 'W GRAND', 'W 127TH', 'W 111TH', 

                                   'N CENTRAL AVE', 'W IRVING PARK', 'N LINCOLN', 

                                   'S CENTRAL AVE', 'S VINCENNES', 'W 79TH', 'N ASHLAND', 

                                   'N OGDEN', 'W BELMONT AVE', 'N MILWAUKEE AVE', 

                                   'N CLYBOURN AVE', 'N RIDGE AVE', 'N BROADWAY', 'W 51ST ST', 

                                   'S JEFFERY', 'W HIGGINS', 'W LAWRENCE', 'N NARRAGANSETT AVE', 

                                   'W CHICAGO AVE', 'S HALSTED', 'S RACINE AVE', 

                                   'W GARFIELD BLVD', 'S INDIANAPOLIS', 'N COLUMBUS DR', 

                                   'W 76th ST', 'E 75TH ST', 'W 55TH', 'W MONTROSE', 

                                   'E ILLINOIS ST', 'S EWING AVE', 'W SUPERIOR ST', 'E 95TH ST', 

                                   'W CERMAK RD', 'N CICERO AVE', 'W DIVISION ST', 

                                   'W BRYN MAWR AVE', 'N NORTHWEST HWY', 'E 87TH ST', 

                                   'E 63RD ST', 'S MARTIN LUTHER KING', 'S ASHLAND AVE', 

                                   'W 83rd ST', 'W 103RD ST', 'W NORTH AVE'], name_value))

one_hot = pd.get_dummies(street_values, 'Title', '_')

df.drop('ADDRESS', axis=1, inplace=True)

df = pd.concat([df, one_hot], axis=1)
df.columns
df[990:1000]
df.drop('Title_', axis=1, inplace=True)
df.columns
#group violations per each day in the data to see the overall change from year to year

date_group = df.groupby(['DATE'])

fig, plot = plt.subplots(figsize=[20, 7])

date_group['VIOLATIONS'].sum().sort_index().plot(color='green')

plot.set_title('Violations on different days', fontsize=25)

plot.tick_params(labelsize='large')

plot.set_ylabel('No of violations', fontsize=20)

plot.set_xlabel('Dates', fontsize=20)
#For the next four types of analysis

#Idea inspired by 

#https://www.kaggle.com/lwkuant/d/chicagopolice/speed-violations/heed-for-speed-the-analysis-you-need



#converting date into time stamp

df['DATE'] = pd.to_datetime(df['DATE'])



#splitting day, month, and year

df['DAY'] = [date.date().strftime('%d') for date in df['DATE']]

df['MONTH'] = [date.date().strftime('%m') for date in df['DATE']]

df['YEAR'] = [date.date().strftime('%Y') for date in df['DATE']]



#adding column with day of week (0=Monday, 6=Sunday)

df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
#Check the year on year changes in violations recorded

year_group = df.groupby(['YEAR'])

fig, plot = plt.subplots(figsize=[10, 7])

year_group['VIOLATIONS'].sum().sort_index().plot(color='blue')

plot.set_title('Violations per year', fontsize=25) 

plot.tick_params(labelsize='large')

plot.set_ylabel('No of violations', fontsize=20)

plot.set_xlabel('Years', fontsize=20)
#Check the monthly changes in violations recorded

month_group = df.groupby(['MONTH'])

fig, plot = plt.subplots(figsize=[10, 7])

month_group['VIOLATIONS'].sum().sort_index().plot(color='blue')

plot.set_title('Violations per month', fontsize=25) 

plot.tick_params(labelsize='large')

plot.set_ylabel('No of violations', fontsize=20)

plot.set_xlabel('Months', fontsize=20)
#group by weekday

weekday_group = df.groupby(['DAY_OF_WEEK'])

fig, plot = plt.subplots(figsize=[15, 7])

weekday_group['VIOLATIONS'].sum().sort_index().plot(color='blue')

plot.set_title('Violations per weekday in general', fontsize=25) 

plot.tick_params(labelsize='large')

plot.set_ylabel('No of violations', fontsize=20)

plot.set_xlabel('Days of week', fontsize=20)
#Check which week days had the most violations each year

years = ['2014','2015','2016']

fig, ax = plt.subplots(1,3, figsize=(22,7), sharey=True)



for i in range(3):

    sns.barplot(df.loc[df['YEAR'] == years[i]]['DAY_OF_WEEK'], 

                df.loc[df['YEAR'] == years[i]]['VIOLATIONS'], 

                errwidth=1, ax=ax[i]);

    ax[i].set_xticklabels(['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'], rotation=25);

    ax[i].set_xlabel(years[i]);

    ax[i].set_ylabel('');
#Check the violations per camera

camera_group = df.groupby(['CAMERA ID']) 

fig, plot = plt.subplots(figsize=[15, 7])

camera_group['VIOLATIONS'].sum().sort_index().plot(color='blue')

plot.set_title('Violations per camera', fontsize=25) 

plot.tick_params(labelsize='large')

plot.set_ylabel('No of violations', fontsize=20)

plot.set_xlabel('Cameras', fontsize=20)
#find out the street with most violations

df['GPS'] = df[['LATITUDE', 'LONGITUDE']].apply(tuple, axis=1)



gps_group = df.groupby(['GPS']) 

fig, axes = plt.subplots(figsize=[20, 7])

gps_group['VIOLATIONS'].sum().sort_index().plot(color='blue')

axes.set_title('Violations per GPS coordinates', fontsize=25)

axes.tick_params(labelsize='large')

axes.set_ylabel('No of violations', fontsize=20)

axes.set_xlabel('GPS coordinates', fontsize=20) 
#group by latitude (is north safer than south?)

latitude_group = df.groupby(['LATITUDE'])

fig, axes = plt.subplots(figsize=[20, 7])

latitude_group['VIOLATIONS'].sum().sort_index().plot(color='blue')

axes.set_title('Violations per latitude', fontsize=25) 

axes.tick_params(labelsize='large')

axes.set_ylabel('No of violations', fontsize=20)

axes.set_xlabel('Latitude', fontsize=20)

axes.set_xlim(41.67, 42.005)
#group by longitude (is west safer than north?)

longitude_group = df.groupby(['LONGITUDE']) 

fig, axes = plt.subplots(figsize=[20, 7])

longitude_group['VIOLATIONS'].sum().sort_index().plot(color='blue')

axes.set_title('Violations per longitude', fontsize=25) 

axes.tick_params(labelsize='large')

axes.set_ylabel('No of violations', fontsize=20)

axes.set_xlabel('Longitude', fontsize=20) 

axes.set_xlim(-87.76, -87.54)