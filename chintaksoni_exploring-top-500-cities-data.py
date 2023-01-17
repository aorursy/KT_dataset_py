# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cities = pd.read_csv('../input/cities_r2.csv')

#get top 5 data rows

cities.head()
# describe data sets

cities.describe()
color = ['red','green','blue','yellow','cyan','pink','orange']

grouping_by_state_population = cities[['population_total','state_name']].groupby('state_name').sum().sort_values(['population_total'],ascending=False)

grouping_by_state_population.plot(kind="bar", color = color) 

plt.xlabel('State Name')

plt.ylabel('Toal Population')

plt.show()
color = ['red','green','blue','yellow','cyan','pink','orange']

effective_literacy_by_state = cities[['effective_literacy_rate_total','state_name']].groupby('state_name').mean().sort_values(['effective_literacy_rate_total'],ascending=False)

effective_literacy_by_state.plot(kind="bar", color = color) 

plt.xlabel('State Name')

plt.ylabel('Literacy Rate')

plt.ylim(75,100)

plt.show()
mizoram = cities[cities['state_name'] == 'MIZORAM']['effective_literacy_rate_total'].mean()

up = cities[cities['state_name'] == 'UTTAR PRADESH']['effective_literacy_rate_total'].mean()

print ("Mizoram Literacy rate :", mizoram)

print ("Uttar Pradesh Literacy rate :", up)
color = ['red','green','blue','yellow','cyan','pink','orange']

effective_literacy_by_state = cities[['effective_literacy_rate_total','name_of_city']].sort_values(['effective_literacy_rate_total'],ascending=False).head(10)

print (effective_literacy_by_state)

effective_literacy_by_state.plot(kind="bar", color = color, x = 'name_of_city', y ='effective_literacy_rate_total') 

plt.xlabel('City Name')

plt.ylabel('Litracy Rate')

plt.ylim(90,100)

plt.show()
color = ['red','green','blue','yellow','cyan','pink','orange']

sex_ration_by_state = cities[['sex_ratio','state_name']].groupby('state_name').mean().sort_values(['sex_ratio'],ascending=False)

sex_ration_by_state.plot(kind="bar", color = color) 

plt.xlabel('State Name')

plt.ylabel('Sex Ratio')

plt.ylim(800,1100)

plt.show()
cities['graduate_percentage'] = (100 * cities['total_graduates'] ) / cities['population_total']

print (cities[['graduate_percentage','name_of_city']].head(10))

graduate_percentage_by_state = cities[['graduate_percentage','state_name']].groupby('state_name').mean().sort_values(['graduate_percentage'],ascending=False)

graduate_percentage_by_state.plot(kind="bar", color = color) 

plt.xlabel('State Name')

plt.ylabel('Graduate Percentage')

plt.ylim(5,30)

plt.show()
nummber_of_city_by_state = cities[['name_of_city','state_name']].groupby('state_name').count().sort_values(['name_of_city'],ascending=False)

nummber_of_city_by_state.plot(kind="bar", color = color) 

plt.xlabel('State Name')

plt.ylabel('Number of Cities')

plt.show()