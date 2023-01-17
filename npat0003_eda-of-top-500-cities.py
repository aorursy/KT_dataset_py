# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as snb



# Inspired from Umesh's analysis, I am just trying out with Python.

#https://www.kaggle.com/umeshnarayanappa/d/zed9941/top-500-indian-cities/exploring-top-500-indian-cities
cities = pd.read_csv('../input/cities_r2.csv')
col = ['yellow', 'green', 'red', 'blue']
# Lets plot the states based on population

population_state= cities[['state_name', 'population_total']].groupby('state_name').sum().sort_values('population_total', ascending=False)

print (population_state)

population_state.plot(kind = 'bar', legend=False, color=col)

plt.show()
# Lets plot the number of cities in each state 

cities_state= cities[['state_name', 'name_of_city']].groupby('state_name').count().sort_values('name_of_city', ascending=False)

print (cities_state)

cities_state.plot(kind = 'bar', legend=False, color=col)

plt.show()
group_city = cities[['state_name', 'name_of_city', 'population_total']].groupby('state_name').count().sort_values('population_total', ascending=False)

group_city.plot(kind = 'bar')

plt.show()
cities['male_per'] = (cities['population_male']/cities['population_total'])*100

grp_male = cities[['male_per', 'state_name']].groupby('state_name').mean().sort_values('male_per', ascending=False)

grp_male.plot(kind='bar', color=col, legend=False)



plt.show()
cities['female_per'] = (cities['population_female']/cities['population_total'])*100

grp_female = cities[['female_per', 'state_name']].groupby('state_name').mean().sort_values('female_per', ascending=False)

grp_female.plot(kind='bar', color=col, legend=False)

plt.show()
cities['grad_per'] = (cities['total_graduates']/cities['population_total'])*100

grad_grp = cities[['state_name', 'grad_per']].groupby('state_name').mean().sort_values('grad_per', ascending=False)

grad_grp.plot(kind='bar', color=col, legend=False)

plt.show()
print (cities.head())
cities['only_male_per'] = (cities.male_graduates/cities.population_male)*100

cities.sort_values('population_total', ascending=False, inplace=True)

cities_top= cities.head(50)

cities_top_per = cities_top[['only_male_per', 'name_of_city']].groupby('name_of_city').mean().sort_values('only_male_per', ascending=False)

print(cities_top_per)

cities_top_per.plot(kind='bar', legend=False, color=col)

plt.show()

#grp_only_male_per = cities[['name_of_city', 'population_total', 'only_male_per']].groupby(['name_of_city']).mean()

#grp_only_male_per.sort_values('population_total', ascending=False)

#grp_only_male_per.plot(kind='bar', color=col, legend=False)

#plt.show()
city_popu_grp = cities[['name_of_city', 'population_total']].groupby('name_of_city').mean().sort_values('population_total', ascending=False).head(50)

city_popu_grp.plot(kind = 'bar', color=col, legend=False)

plt.show()
cities.head()
popu_top = cities.sort_values('population_total', ascending=False)

popu_top = popu_top.head(50)

popu_top['male_per'] = (popu_top['population_male']/popu_top['population_total'])*100

male_popu_top = popu_top[['male_per', 'name_of_city']].groupby('name_of_city').mean().sort_values('male_per', ascending=False)

male_popu_top.plot(kind='bar', legend=False, color=col)

plt.show()
popu_top = cities.sort_values('population_total', ascending=False)

popu_top = popu_top.head(50)

popu_top['female_per'] = (popu_top['population_female']/popu_top['population_total'])*100

female_popu_top = popu_top[['female_per', 'name_of_city']].groupby('name_of_city').mean().sort_values('female_per', ascending=False)

female_popu_top.plot(kind='bar', legend=False, color=col)

plt.show()
popu_top['tot_grad_per'] = ((popu_top['male_graduates']+ popu_top['female_graduates'])/popu_top['population_total'])*100

grad_tot_per= popu_top[['tot_grad_per', 'name_of_city']].groupby('name_of_city').mean().sort_values('tot_grad_per', ascending = False)

grad_tot_per.plot(kind='bar', legend=False, color=col)

plt.show()