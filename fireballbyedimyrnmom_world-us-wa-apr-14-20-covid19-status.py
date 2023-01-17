#Libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt #plotting, math, stats

%matplotlib inline

import seaborn as sns #plotting, regressions
#Dataset from the World Health Organization

World = pd.read_csv("../input/httpsourworldindataorgcoronavirussourcedata/apr13update.csv")

plt.figure(figsize=(21,8)) # Figure size

World.groupby("location")['total_cases'].max().plot(kind='bar', color='darkkhaki')
World.head(2)
##For ease of visualization

China=World.loc[World['location']== 'China']

USA=World.loc[World['location']== 'United States']

Ital=World.loc[World['location']== 'Italy']

SK=World.loc[World['location']== 'South Korea']

Brzl=World.loc[World['location']== 'Brazil']
some1=pd.concat([USA, China, Ital, SK, Brzl]) 



some1=some1.sort_values(by=['date'], ascending=False)

some1.head(2)
##Cases in some countries

plt.figure(figsize=(16,7))

sns.lineplot(x="date", y="total_cases", hue="location",data=some1)

plt.title('Cases per day') # Title

plt.xticks(some1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
##Death rates in some countries

plt.figure(figsize=(16,7))

sns.lineplot(x="date", y="total_deaths", hue="location",data=some1)

plt.title('Deaths per day') # Title

plt.xticks(some1.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()
US = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

US=US.drop(['fips'], axis = 1) 
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='cases', data=US, marker='o', color='darkseagreen') 

plt.title('Cases per day across the USA') # Title

plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees

plt.show()
#The numbers are exponential (the total from the previous day + that day's data)

US.sort_values(by=['cases'], ascending=False).head(10)
#Cases across states 

plt.figure(figsize=(19,7)) # Figure size

US.groupby("state")['cases'].max().plot(kind='bar', color='sienna')

plt.title('States COVID-19 case rate') # Title
#DEATH rates

plt.figure(figsize=(19,7)) 

US.groupby("state")['deaths'].max().plot(kind='bar', color='mediumvioletred')

plt.title('States COVID-19 death rate') 
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='deaths', data=US, marker='o', color='dimgrey') 

plt.title('Deaths across the USA') # Title

plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees

plt.show()
WA=US.loc[US['state']== 'Washington']
WA.sort_values(by=['cases'], ascending=False).head(10)
plt.figure(figsize=(12,6)) # Figure size

WA.groupby("county")['cases'].max().plot(kind='bar', color='goldenrod')

plt.title('Total cases across WA counties') 
plt.figure(figsize=(12,6)) # Figure size

WA.groupby("county")['deaths'].max().plot(kind='bar', color='lightcoral')

plt.title('Deaths total across WA counties') 
plt.figure(figsize=(16,12))

sns.lineplot(x="date", y="deaths", hue="county",data=WA)

plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()