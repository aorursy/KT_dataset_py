import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm

#import squarify

%matplotlib inline
data = pd.read_csv('../input/cities_r2.csv')

data.head()
# Plot states with most cities

states = data[['name_of_city', 'state_name']].groupby('state_name').count().sort_values('name_of_city', ascending=False)

plt.figure(figsize=(8, 5))

ax = sns.barplot(x=states.index, y=states['name_of_city'], palette= sns.color_palette("muted"))

ax.set_ylabel("Number Of Cities")

ax.set_xlabel("States")

for item in ax.get_xticklabels():

    item.set_rotation(90)
plt.figure(figsize=(5, 5))

sns.heatmap(states, annot=True, fmt="d", linewidths=.5)
from matplotlib.ticker import FuncFormatter



formatter = FuncFormatter(lambda x, pos:'{} mil'.format(int(x*1e-6)))

plt.figure(figsize=(8, 5))

states = data[['population_total', 'state_name']].groupby('state_name').sum().sort_values('population_total',ascending=False)

ax = sns.barplot(x=states.index, y=states['population_total'], palette= sns.color_palette("muted"))

ax.yaxis.set_major_formatter(formatter)

ax.set_ylabel("Total Poulation")

ax.set_xlabel("States")

for item in ax.get_xticklabels():

    item.set_rotation(90)
states = data[['literates_male', 'literates_female', 

               'state_name']].groupby('state_name').sum().sort_values(['literates_male', 'literates_female'], ascending=False)

plt.figure(figsize=(8, 5))

sns.set_palette(sns.color_palette("muted"))

plt.xticks(range(len(states.index)), list(states.index), rotation=90)

plt.plot(states['literates_male'].values, ls="--")

plt.plot(states['literates_female'].values)
literacy_rate = (data['literates_total']/data['population_total'])*100

fig, ax = plt.subplots()

fig.set_size_inches(8, 5)

sns.set_palette(sns.color_palette("muted"))

ax.scatter(literacy_rate.values, data['sex_ratio'].values)

for idx, txt in enumerate(list(data['state_name'])):

    ax.annotate(txt,(literacy_rate.values[idx], data['sex_ratio'].values[idx]), size=3)
corr = data[['0-6_population_total', 'effective_literacy_rate_total', 'literates_total',

            'population_total', 'sex_ratio', 'total_graduates']].corr()

plt.figure(figsize=(8, 5))

sns.heatmap(corr, annot=True, linewidths=.5)
#Cant Import module squarify otherwise it renders a tree plot for states according to population

#states = data[['population_total','state_name']].groupby('state_name').sum().sort_values(

#                ['population_total'], ascending=False)

#fig = plt.figure(figsize=(8, 10))

# sns.set_palette(sns.color_palette("muted"))

#fig.suptitle("States By Total Population", fontsize=20)

#ax = fig.add_subplot(111, aspect="equal")

#ax = squarify.plot(states['population_total'], label=states.index, ax=ax, alpha=.7, color=sns.color_palette("muted", 8))

#ax.set_xticks([])

#ax.set_yticks([])