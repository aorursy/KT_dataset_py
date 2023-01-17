#Import necessaries libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import missingno as msno



#read the data

data = pd.read_csv('../input/data-police-shootings/fatal-police-shootings-data.csv')
#examine first few rows

data.head()
# How many rows and columns?

data.shape
# Check for null values

data.isna().sum()
# Visualize missing values as matrix

msno.matrix(data)
# Drop all missing values

data.dropna(inplace = True)



data.isna().sum()
# Add Year and month column

data['Year'] = pd.to_datetime(data['date']).dt.year

data['Month']= pd.to_datetime(data['date']).dt.month



data

data.info()
sns.distplot(data['age']).set_title('Most Victims by Age')
# Male or Female victim the most

sns.countplot(x = 'gender', data = data).set_title('Most Victims by Gender')

# Shot by race

by_race = data[data['manner_of_death']== 'shot']

sns.countplot(data = by_race, x = 'race' ).set_title('Most Shot Victims by Race')

# Does victims has any signs of mental illness



sns.countplot(x = 'signs_of_mental_illness', data = data).set_title('Victims by Mental illness')
# shootings over year

sns.countplot(x = 'Year',data = data).set_title('Victims From Year 2015 -2020')
# Top 10 states with most shootings

topten_state = data['state'].value_counts()[:10]

topten_state = pd.DataFrame(topten_state).reset_index()





ax = sns.barplot(data = topten_state, x = 'index', y = 'state')

ax.set(xlabel = 'State',ylabel = 'No of Death', title = 'Top Ten States with Shooting Deaths')



# Does some month have more shooting than others

monthly_shooting = data[['Month', 'manner_of_death']]

monthly_shooting['deaths'] = 1

monthly_death= monthly_shooting.groupby('Month').sum()

monthly_death = monthly_death.reset_index()









sns.barplot(data = monthly_death, x = 'Month',y = 'deaths').set_title('Month-Wise Number of Deaths')
# Top 10 states with most shootings

topfive_city = data['city'].value_counts()[:5]

topfive_city = pd.DataFrame(topfive_city).reset_index()



ax = sns.barplot(data = topfive_city , x = 'index', y = 'city')

ax.set(xlabel = 'City',ylabel = 'No of Death', title = 'Top Five City with Shooting Deaths')
