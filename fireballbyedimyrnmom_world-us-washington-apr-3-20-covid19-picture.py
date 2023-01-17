import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt #plotting, math, stats

%matplotlib inline

import seaborn as sns #plotting, regressions
#Dataset from the World Health Organization

World = pd.read_csv("../input/httpsourworldindataorgcoronavirussourcedata/full_data(12).csv")

plt.figure(figsize=(21,8)) # Figure size

World.groupby("location")['total_cases'].max().plot(kind='bar', color='darkgreen')
#total deaths worldwide

plt.figure(figsize=(22,10)) # Figure size

World.groupby("location")['total_deaths'].max().plot(kind='bar', color='tan')
US = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

US=US.drop(['fips'], axis = 1) 
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='cases', data=US, marker='o', color='indigo') 

plt.title('Cases per day in the USA') # Title

plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees

plt.show()
US.sort_values(by=['cases'], ascending=False).head(30)
#total deaths worldwide

plt.figure(figsize=(19,7)) # Figure size

US.groupby("state")['cases'].max().plot(kind='bar', color='darkblue')
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='deaths', data=US, marker='o', color='dimgrey') 

plt.title('Deaths per day in the USA') # Title

plt.xticks(US.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees

plt.show()
#total deaths worldwide

plt.figure(figsize=(19,7)) # Figure size

US.groupby("state")['deaths'].max().plot(kind='bar', color='coral')
WA=US.loc[US['state']== 'Washington']
WA.sort_values(by=['cases'], ascending=False).head(30)
plt.figure(figsize=(12,8)) # Figure size

WA.groupby("county")['cases'].max().plot(kind='bar', color='teal')
plt.figure(figsize=(12,8)) # Figure size

WA.groupby("county")['deaths'].max().plot(kind='bar', color='goldenrod')
plt.figure(figsize=(16,11))

sns.lineplot(x="date", y="deaths", hue="county",data=WA)

plt.xticks(WA.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()