#some standard imports

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import matplotlib.dates as mdates

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime
#what are the data files?

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# this is the NY county-by-county testing data from NY State

ny = pd.read_csv('/kaggle/input/New_York_State_Statewide_COVID-19_Testing.csv', delimiter=',')

ny.dataframeName = 'New_York_State_Statewide_COVID-19_Testing.csv'

# Fix format of Test Date column

ny['Test Date']= pd.to_datetime(ny['Test Date']) 

nRow, nCol = ny.shape

print(f'There are {nRow} rows and {nCol} columns')

ny.head(10)
#I also found some county-by-county population data.  It's based on census data, with estimates since the last census

pops = pd.read_csv('/kaggle/input/Annual_Population_Estimates_for_New_York_State_and_Counties__Beginning_1970.csv', delimiter=',')

pops.head(5)
#Delete NY State (aggregate) from the population data. For some reason they included this alongside all the counties

pops=pops[pops.Geography != 'New York State']

#Keep only the 2019 populations

pops_2019 = pops[pops['Year']==2019]

#take the top 10 counties by population

top_counties=pops_2019.nlargest(10,'Population').sort_values('Population', ascending=False)

#Remove the word 'County' so we can join to the test data set, which doesn't have this word

top_counties['Geography'] = top_counties['Geography'].str.replace(r' County', '')

top_counties_list = top_counties['Geography'].to_list()

#Change name of column for consistency, and drop some other fields

top_counties.rename(columns = {'Geography':'County'}, inplace = True)

top_counties.drop(['FIPS Code', 'Year', 'Program Type'], axis=1, inplace=True)

#This is the list of top counties by population

top_counties
#calculate the percents of tests performed, and positive results, based on the population of each county

#it's possible some people had multiple tests, but ignore that

df = ny[ny.County.isin(top_counties_list)]

df1 = pd.merge(df,top_counties,on = 'County')

df1['Cum Percent Tested']=100*df1['Cumulative Number of Tests Performed']/df1['Population']

df1['Cum Percent Positives']=100*df1['Cumulative Number of Positives']/df1['Population']

df1
graph = df1.pivot(index='Test Date', columns='County', values='Cumulative Number of Tests Performed')

graph
top_counties_list = graph.columns.to_list()



plt.figure(figsize=(12,12))

plt.plot(graph)

plt.title('Cumulative Number of Tests Performed')

plt.xlabel('Date')

plt.ylabel('Cumulative Tests')

plt.legend(top_counties_list)

plt.show()
graph=df1.pivot(index='Test Date', columns='County', values='Cum Percent Positives')



top_counties_list = graph.columns.to_list()



plt.figure(figsize=(12,12))

plt.plot(graph)

plt.title('Cumulative Positive Tests as a Percent of Population')

plt.xlabel('Date')

plt.ylabel('Cumulative Tests')

plt.legend(top_counties_list)

plt.show()

graph = df1.pivot(index='Test Date', columns='County', values='Cum Percent Tested')



top_counties_list = graph.columns.to_list()



plt.figure(figsize=(12,12))

plt.plot(graph)

plt.title('Cumulative Percent of Population Tested')

plt.xlabel('Date')

plt.ylabel('Cumulative Percent Tested')

plt.legend(top_counties_list)

plt.show()

# read in the worldwide data

world = pd.read_csv('/kaggle/input/full-list-cumulative-total-tests-per-thousand.csv', delimiter=',')

# Fix format of Test Date column

world['Date']= pd.to_datetime(world['Date']) 

nRow, nCol = world.shape

print(f'There are {nRow} rows and {nCol} columns')

world
#I'm going to ignore entries without test data

world = world.dropna()

#let's look at a subset of countries

country_list=['South Korea', 'United States', 'Canada', 'Italy', 'Australia','New Zealand']

c=world[world['Entity'].isin(country_list)]

c['Cum Percent Tested']=c['Total tests per thousand']/10.0

#sort by country

c.Entity = c.Entity.astype("category")

c.Entity.cat.set_categories(country_list, inplace=True)

#pd.set_option('display.max_rows', None)

c
graph = c.pivot(index='Date', columns='Entity', values='Cum Percent Tested')

plt.figure(figsize=(12,12))

plt.plot(graph)

plt.title('Cumulative Percent of Population Tested')

plt.xlabel('Date')

plt.ylabel('Cumulative Percent Tested')

plt.legend(country_list)

plt.show()
# read in the worldwide data

tests_per_thous = pd.read_csv('/kaggle/input/daily-covid-19-tests-per-thousand-rolling-3-day-average.csv', delimiter=',')

#I'm going to ignore entries without test data

#world = world.dropna()

# Fix format of Test Date column

tests_per_thous['Date']= pd.to_datetime(tests_per_thous['Date']) 

tests_per_thous
country_list=['South Korea', 'United States', 'Canada', 'Italy', 'Australia','New Zealand']

c=tests_per_thous[(tests_per_thous['Entity'].isin(country_list)) & (tests_per_thous['Date']>pd.Timestamp(2020, 4, 1)) ]

#sort by country

c.Entity = c.Entity.astype("category")

c.Entity.cat.set_categories(country_list, inplace=True)

c = c.reset_index()

del c['index']

c
graph = c.pivot(index='Date', columns='Entity', values='3-day rolling mean of daily change in total tests per thousand')

plt.figure(figsize=(12,12))

plt.plot(graph)

plt.title('Daily Tests Performed Per Thousand Population')

plt.xlabel('Date')

plt.ylabel('3-day rolling mean of daily change in total tests per thousand')

plt.legend(country_list)

plt.show()
