import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
fleet = pd.read_csv('../input/Fleet Data.csv')
fleet.head()
fleet.describe()
fleet.info()
fleet['Parent Airline'].nunique()
fleet['Airline'].nunique()
fleet['Aircraft Type'].nunique()
fleet.columns
#Separating quantity and cost data from age:
aircraftfleet = fleet[['Airline','Aircraft Type', 'Current', 'Future', 'Historic', 'Total', 'Orders', 'Total Cost (Current)']]
parentfleet = fleet[['Parent Airline','Aircraft Type', 'Current', 'Future', 'Historic', 'Total', 'Orders', 'Total Cost (Current)']]
aircraftfleet.head(5)
parentfleet.head(5)
# Grouping data by Airline name to see tha aircraft fleet size including all aircraft type:
aircraftfleet.groupby(axis=0, by='Airline').sum().head(10)
# Top 20 Parent Airlines with the biggest currently active aircraft fleet:
aircraftfleet.groupby(by='Airline').sum()['Current'].sort_values(ascending=False).head(20)
# Grouping data by Parent Airline name to see tha aircraft fleet size including all aircraft type:
parentfleet.groupby(axis=0, by='Parent Airline').sum().head(10)
# Top20 Airlines with the biggest currently active aircraft fleet:
parentfleet.groupby(by='Parent Airline').sum()['Current'].sort_values(ascending=False).head(50)
# Fleet size for these specific airlines:
fleet[(fleet['Parent Airline'] == 'Air France/KLM') | (fleet['Parent Airline'] == 'Aeroflot') | (fleet['Parent Airline'] == 'Lufthansa') | (fleet['Parent Airline'] == 'Emirates') | (fleet['Parent Airline'] == 'American Airlines')].groupby(by='Parent Airline').sum()['Current'].sort_values(ascending=False).head(50)
# selected_airlines = fleet[(fleet['Parent Airline'] == 'Air France/KLM') | (fleet['Parent Airline'] == 'Aeroflot') | (fleet['Parent Airline'] == 'Lufthansa') | (fleet['Parent Airline'] == 'Emirates') | (fleet['Parent Airline'] == 'American Airlines')]
selected_airlines = fleet[(fleet['Parent Airline'] == 'Air France/KLM') | (fleet['Parent Airline'] == 'Aeroflot') | (fleet['Parent Airline'] == 'Lufthansa') | (fleet['Parent Airline'] == 'Emirates') | (fleet['Parent Airline'] == 'American Airlines')].copy()
# Top20 oldest currently active aircraft types based on the average age:
selected_airlines[['Parent Airline', 'Aircraft Type','Average Age']].dropna(axis=0).sort_values(by='Average Age', ascending=False).head(20)
# Top20 newest currently active planes:
selected_airlines[['Parent Airline', 'Aircraft Type','Average Age']].dropna(axis=0).sort_values(by='Average Age').head(20)
selected_airlines.columns
plt.figure(figsize=(14,10))
sns.countplot(data=selected_airlines, x='Parent Airline', hue='Airline')
plt.legend(bbox_to_anchor=(1, 1.0))
# Number of unique aircraft types
selected_airlines['Aircraft Type'].nunique()
# Here we can find the number of unique aircraft types used by Emirates:
selected_airlines[selected_airlines['Parent Airline'] == 'Emirates']['Aircraft Type'].nunique()
selected_airlines[selected_airlines['Parent Airline'] == 'Emirates'][['Aircraft Type', 'Current', 'Future',
       'Historic', 'Total', 'Orders']]
sns.set_style('darkgrid')
plt.figure(figsize=(14,10))
sns.boxplot(data=selected_airlines, x='Parent Airline', y='Average Age', palette='coolwarm')
avg = selected_airlines.dropna(axis=0, subset=['Average Age',])[['Parent Airline','Airline','Aircraft Type','Average Age']]
# List of unique airplanes for these airlines:
avg['Aircraft Type'].unique()
plt.figure(figsize=(14,10))
sns.boxplot(data=avg, y='Aircraft Type', x='Average Age')
biggies = selected_airlines[(selected_airlines['Aircraft Type'] == 'Airbus A380') | (selected_airlines['Aircraft Type'] == 'Airbus A330') | (selected_airlines['Aircraft Type'] == 'Airbus A340') | (selected_airlines['Aircraft Type'] == 'Boeing 747') | (selected_airlines['Aircraft Type'] == 'Boeing 777') | (selected_airlines['Aircraft Type'] == 'Boeing 787 Dreamliner')]
biggies.head(5)
biggies.sort_values('Aircraft Type')[biggies['Current'] > 0][['Parent Airline', 'Airline', 'Aircraft Type', 'Current']].head(20)
plt.figure(figsize=(10,6))
sns.countplot(data=biggies, x='Aircraft Type', hue='Parent Airline')
plt.legend(bbox_to_anchor=(1, 1.0))

# The plot demonstrates how many Airlines under major daughter airlines use big aircraft.
selected_airlines.columns
bigsorted = biggies.drop(axis=1, columns='Average Age').groupby('Aircraft Type').sum()
bigsorted.head(10).sort_values('Current', ascending=False)

# As we can see the most used type of Aircraft among the selected Airlines is Boeing 777.
# Boeing 787 Dreamliner is quite a new plane and is not that widely used yet.
# Boeing 747 is quite old already and probably most airlines will soone replace these planes with newer ones, like 787.
bigsorted = biggies.groupby('Aircraft Type').mean().sort_values('Average Age', ascending=False)
bigsorted['Average Age']

# As we can see below Boeing 747 is indeed in general older that other planes, while Boeing 787 planes are the youngest ones.
airplanes = biggies[['Parent Airline', 'Aircraft Type', 'Current']].copy()
airplanes.dropna(axis=0, subset=['Current',], inplace=True)
airplanes.sort_values('Aircraft Type')
airplanes = airplanes.groupby(by=['Parent Airline', 'Aircraft Type']).sum()
airplanes = airplanes.reset_index()
sns.lmplot(x='Parent Airline', y='Current', hue='Aircraft Type', data=airplanes, fit_reg=False, size=6)
g = sns.FacetGrid(airplanes, row='Parent Airline' , col="Aircraft Type", hue='Aircraft Type', margin_titles=True)
g = g.map(plt.bar, "Aircraft Type", "Current")

# Facit Grid plot to display data abour Airline fleet in details. Columns - Aircraft Type, Rows - Airline
plt.figure(figsize=(10,6))
sns.barplot(data=airplanes, x='Parent Airline',y='Current', hue='Aircraft Type')
plt.legend(bbox_to_anchor=(1, 1.0))

# ALternative barplot graph to make it easier to compare the fleets of different Airlines.
