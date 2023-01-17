# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/cost-of-living/cost-of-living.csv')
data = data.rename(columns={"Unnamed: 0": "food"})
data
def student_spending_weekly(data,place):

    spending = 0

    food_dic = {}

    k = -1

    x = ""

    for name in data.food:

        k += 1

        if k == 21:

            x = random.choice(['Apartment (1 bedroom) in City Centre','Apartment (1 bedroom) Outside of Centre','Apartment (3 bedrooms) in City Centre','Apartment (3 bedrooms) Outside of Centre'])

        if x == name:

            food_dic[x] = 1 

        if k > 51:

            food_dic[name] = 1 

        food_dic[name] = random.randint(0,10)

    for index,row in data.iterrows():

        spending += row[place] * food_dic[row['food']]

        

    return spending
student_spending_weekly(data,'Helsinki, Finland')
the_weekly_spending = []

for name in data.columns:

    if name == 'food':

        pass

    else:

        the_weekly_spending.append(student_spending_weekly(data,str(name)))

    
cities = data.columns.drop('food')
cities_spending_df = pd.DataFrame()
cities_spending_df['cities'] = cities

cities_spending_df['weekly_spending'] = the_weekly_spending
cities_spending_df = cities_spending_df.sort_values('weekly_spending',ascending =False)
import seaborn as sns

from matplotlib import pyplot

fig, ax = pyplot.subplots(figsize=(20,15))

ax = sns.barplot(x='weekly_spending', y='cities',data = cities_spending_df.head(30))
cities_spending_df = cities_spending_df.sort_values('weekly_spending',ascending =True)
import seaborn as sns

from matplotlib import pyplot

fig, ax = pyplot.subplots(figsize=(20,15))

ax = sns.barplot(x='weekly_spending', y='cities',data = cities_spending_df.head(30))