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
import os

os.getcwd()

cities = pd.read_csv("../input/cities_r2.csv")
cities.head(5)
cities.info()

## -> No null values present
cities.shape
# List no. of unique states

print(cities['state_name'].unique())

print(len(cities['state_name'].unique()))



print(cities['dist_code'].unique())

print(len(cities['dist_code'].unique()))
# Population at state level

population_state = cities[['state_name','population_total']].groupby('state_name').sum().sort_values('population_total',ascending=False)

print(population_state)
colr = ['red','green','yellow','blue']

population_state.plot(kind='bar',color=colr)
# Top 5 cities with highest female population

cities[['name_of_city','population_female']].sort_values(by='population_female',ascending=False).head(5)

# -> Its Mumbai and Delhi at top
# Top 5 states with highest literacy rate 

## Pending if any one can help would be great 

#cities[['state_name','effective_literacy_rate_female']].groupby('state_name').sum().apply()