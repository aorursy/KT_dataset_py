import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

sns.set(color_codes=True)
animals = pd.read_csv('/kaggle/input/domesticated-animals-in-mongolia-19702017/Animal Numbers.csv', parse_dates=["Year"])

animals.rename(columns={'Type of livestock':'Animal','Heads (Thousands)':'Heads'}, inplace=True)

animals.head(10)
animals['Year'] = animals['Year'].dt.year
animals.describe()
ax = sns.countplot(x="Animal", data=animals)

for p in ax.patches:

    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.35, p.get_height()+1))
sns.set(rc={'figure.figsize':(12,6)})

sns.scatterplot(x='Year',y='Heads', hue = 'Animal', data=animals)

g = sns.FacetGrid(animals, col="Animal", margin_titles=True)

g.map(sns.regplot, "Year", "Heads",fit_reg=False, x_jitter=.5)
from numpy import median

sns.barplot(x='Heads',y='Animal', data=animals, palette='Spectral')
sns.catplot(x='Heads',y='Animal', data=animals, palette='Spectral', col="Year", height=3, kind="bar", col_wrap=6)