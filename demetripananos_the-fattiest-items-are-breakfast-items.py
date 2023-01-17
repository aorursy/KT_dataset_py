import numpy as np

import seaborn.apionly as sns

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv('../input/menu.csv')
df['percent fat'] = 100*df['Calories from Fat']/df.Calories



g = df[['Item','percent fat']].sort_values('percent fat', ascending = False)



g = g[g['percent fat']>50]





fig, ax = plt.subplots(figsize=(5,15))

sns.stripplot(ax = ax,data = g, x = 'percent fat', y = 'Item', s = 10, color = 'grey')



ax.yaxis.grid(True)



ax.set_xlabel('Percent Calories from Fat')


order = df.groupby('Category').agg({'percent fat':'mean'}).sort_values('percent fat', ascending = False).index



fig, ax = plt.subplots(figsize=(8,5))



sns.factorplot(data = df, x = 'Category',y = 'percent fat', ax = ax, order = order)



ax.set_xticklabels(ax.get_xticklabels(),rotation = 45)



ax.set_ylabel('Mean Percent Calories from Fat')

