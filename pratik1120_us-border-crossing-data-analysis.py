import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# importing important libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#summoning the data demon
data = pd.read_csv('../input/us-border-crossing-data/Border_Crossing_Entry_Data.csv')
data.head(2)
# let's check the shape to make an estimate of how big the data is
data.shape
#let's check the decription of the data
data.describe(include='all')
# lets take a quick loot at the columns
data.columns
#making a copy 
df = data.copy()

#lets look at the data
df.head()
#lets check unique number of ports in the data
print(df['Port Name'].nunique())
#let's check which port has highest frequency
df['Port Name'].value_counts()
#converting Date to datetime instance
df['Date'] = pd.to_datetime(df['Date'])

#checking the numbr of unique dates
df['Date'].nunique()
#checking the beginning and end of data
df['Date']
df['Date'].dt.year.value_counts()
# storing all unique years in a list
# ignoring 2020 as it have only 2 months of data
years = df['Date'].dt.year.unique().tolist()
years.remove(2020)
type(years)
df2 = df['Port Name'].value_counts().reset_index()
df2.loc[0]
#creating two lists
counts = [] #counts of the busiest ports
busiest_ports = [] #busiest ports of the year
for year in years:
    df1 = df[df['Date'].dt.year==year]
    df1 = df1['Port Name'].value_counts().reset_index()
    busiest_ports.append(df1.loc[0]['index'])
    counts.append(df1.loc[0]['Port Name'])
    
fig = plt.figure(figsize=(15,8))
sns.barplot(years, counts)
plt.show()
busiest_ports[:20]
fig = plt.figure(figsize=(28,30))
for year,num in zip(years, range(1,25)):
    df1 = df[df['Date'].dt.year==year]
    df1 = df1['Port Name'].value_counts().reset_index()
    ax = fig.add_subplot(8,3,num)
    ax.bar(df1.loc[:4]['index'], df1.loc[:4]['Port Name'])
    ax.set_title(year)
#getting a new copy of the data
df = data.copy()

#lets see the data
df.head(2)
#lets see the unique number of states in the dataset
df['State'].nunique()
#converting date to datetime instance
df['Date'] = pd.to_datetime(df['Date'])
# creating a separate dataframe for 2020 data
df_2020 = df[df['Date'].dt.year==2020]
# checking unique states in 2020
df_2020['State'].unique()
df_2020['State'].value_counts()
df1 = df_2020['State'].value_counts().reset_index()
fig = plt.figure(figsize=(10,5))
barlist = plt.bar(df1.loc[:4]['index'], df1.loc[:4]['State'])
barlist[0].set_color('m')
barlist[1].set_color('b')
barlist[2].set_color('g')
barlist[3].set_color('c')
barlist[4].set_color('y')
plt.plot(df1.loc[:4]['index'], df1.loc[:4]['State'], c='red',linewidth=7.0)
plt.title('Top 5 states with highest cossings')
plt.show()
data.head(3)
#change to datetimg
data['Date'] = pd.to_datetime(data['Date'])

#separating 2019 data
df_2019 = data[data['Date'].dt.year==2019]

#checking most used border
df_2019['Border'].value_counts()
borders = df_2019['Border'].value_counts().reset_index()

fig = plt.figure(figsize=(10,5))
sns.barplot(borders['index'], borders['Border'])
plt.xlabel('Borders')
plt.ylabel('Number of crossing')
plt.title('Border vs crossing in 2019')
plt.show()
#getting unique years
years = data['Date'].dt.year.unique().tolist()

#getting border data for each year
canada = []
mexico = []
for year in years:
    df = data[data['Date'].dt.year==year]
    borders = df['Border'].value_counts().reset_index()
    canada.append(int(borders.loc[borders['index']=='US-Canada Border']['Border']))
    mexico.append(int(borders.loc[borders['index']=='US-Mexico Border']['Border']))
len(canada), len(mexico)
years
fig, ax = plt.subplots(figsize=(20,7))
x = np.arange(len(years))
width = 0.35
ax.bar(x-width/2, canada, width, label='US-Canada Border')
ax.bar(x+width/2, mexico, width, label='US-Mexico Border')

ax.set_ylabel('Crossing counts')
ax.set_title('Year wise border data')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()
data.head(3)
#separating 2019 data
df_2019 = data[data['Date'].dt.year==2019]

quars = []
for i in range(1,12,3):
    quars.append(df_2019[(df_2019['Date'].dt.month == i)|(df_2019['Date'].dt.month == i+1)|(df_2019['Date'].dt.month == i+2)].shape[0])
    
len(quars)
fig = plt.figure(figsize=(10,7))
plt.plot(['1st quar', '2nd quar','3rd quar', '4th quar'], quars)
plt.show()
#getting unique years
years = data['Date'].dt.year.unique().tolist()

#getting border data for each year
first = []
second = []
third = []
fourth = []
for year in years:
    df = data[data['Date'].dt.year==year]
    borders = df['Border'].value_counts().reset_index()
    first.append(df[(df['Date'].dt.month == 1)|(df['Date'].dt.month == 2)|(df['Date'].dt.month == 3)].shape[0])
    second.append(df[(df['Date'].dt.month == 4)|(df['Date'].dt.month == 5)|(df['Date'].dt.month == 6)].shape[0])
    third.append(df[(df['Date'].dt.month == 7)|(df['Date'].dt.month == 8)|(df['Date'].dt.month == 9)].shape[0])
    fourth.append(df[(df['Date'].dt.month == 10)|(df['Date'].dt.month == 11)|(df['Date'].dt.month == 12)].shape[0])
fig, ax = plt.subplots(figsize=(20,7))
x = np.arange(len(years))
width = 0.2
ax.bar(x-(width*3)/2, first, width, label='First Quarter')
ax.bar(x-width/2, second, width, label='Second Quarter')
ax.bar(x+width/2, third, width, label='Third Quarter')
ax.bar(x+(width*3)/2, fourth, width, label='Fourth Quarter')

ax.set_ylabel('Crossing counts')
ax.set_xlabel('Years')
ax.set_title('Quarter wise crossing data')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()
#checking all unique values in the 'Measure' column
data['Measure'].unique()
#creating table for vehicle count
df = data['Measure'].value_counts().reset_index()

fig = plt.figure(figsize=(20,7))
sns.barplot(df['Measure'], df['index'])
plt.title('Vehicles used for crossing border')
plt.show()
# checking the sum of values for each vehicle
vehicles = data['Measure'].unique()
values = []

for vehicle in vehicles:
    df = data[data['Measure']==vehicle]
    values.append(df['Value'].mean()) #if you wonder why I took mean then check the below markdown
fig = plt.figure(figsize=(10,10))
sns.barplot(values, vehicles)
plt.title('Value by each vehicle')
plt.show()
data.head(2)
#separating on countries
countries = ['Canada', 'Mexico']
values = []

for country in countries:
    df = data[data['Border']=='US-{} Border'.format(country)]
    values.append(df['Value'].mean())
fig = plt.figure(figsize=(10,5))
sns.barplot(values, countries)
plt.title('Country vs Value')
plt.xlabel('Number of People crossing US border from this country')
plt.show()
