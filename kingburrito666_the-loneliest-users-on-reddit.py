import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')
data = pd.read_csv('../input/foreveralone.csv')

data.head(4)
male_data = data[data.gender == 'Male']

female_data = data[data.gender == 'Female']



plt.figure(figsize=(8, 8))

plt.hist([male_data.age, female_data.age] , alpha=0.5, bins=20, label=['Male', 'Female'])

plt.legend(loc='upper right')



plt.title("Distribution of users' ages. M/F")

plt.ylabel('count of users')

plt.xlabel('age');

plt.show()
by_race = data.groupby('race').size().sort_values(ascending=False).rename('counts').head(9)

labels = ['White non-Hispanic',

    'Asian',

    'Hispanic (of any race)',

    'Black',

    'Mixed',

    'Indian',

    'Middle Eastern',

    'Multi',

    'European']

bomb = (0,0,0,0,0.1,0.1,0.1,0.1,0.1)

fig, ax = plt.subplots(figsize=(8,8))

ax.pie(by_race, explode=bomb, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')

ax.axis('equal')

plt.show()
from matplotlib.gridspec import GridSpec



by_depre = data.groupby('depressed').size()

by_suici = data.groupby('attempt_suicide').size()

by_socia = data.groupby('social_fear').size()

by_virgin = data.groupby('virgin').size()



plt.figure(figsize=(8, 8))

the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 0], aspect=1)

plt.pie(by_depre, labels=['Not Depressed', 'Depressed'], autopct='%1.1f%%', shadow=True)



plt.subplot(the_grid[0, 1], aspect=1)

plt.pie(by_suici, labels=['Not Suicidal', 'Suicidal'], autopct='%1.1f%%', shadow=True)



plt.subplot(the_grid[1, 0], aspect=1)

plt.pie(by_socia, labels=['Fear', 'No fear'], autopct='%1.1f%%', shadow=True)



plt.subplot(the_grid[1, 1], aspect=1)

plt.pie(by_virgin, labels=['Virgin', 'Not Virgin'], autopct='%1.1f%%', shadow=True)

plt.show()