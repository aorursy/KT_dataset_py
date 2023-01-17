# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('/kaggle/input/honey-production/honeyproduction.csv')
df.info()
df.head()
## Scatter plot depicts the joint distribution of two variables using a cloud of points, 

## where each point represents an observation in the dataset. This is very useful in finding any

## substantial relationship between 2-variables if any. So for any continuous variables, this should

## be your first choice of visualization

plt.figure(figsize=(10,4))

sns.scatterplot(x='priceperlb', y='totalprod', data=df)
##  In case you want to plot scatter plot between multiple variables, one can use pairplot

sns.pairplot(df[['numcol', 'yieldpercol', 'totalprod', 'stocks', 'priceperlb', 'prodvalue']])
sns.jointplot(x='numcol', y='totalprod', data=df, kind='reg')
##With some datasets, you may want to understand changes in one variable as a function of time, 

## or a similarly continuous variable. In this situation, a good choice is to draw a line plot

## In below one can clerly understand that there is decrease in production every consecutive year

sns.lineplot(x='year', y='totalprod', data=df)
plt.figure(figsize=(10,4))

sns.stripplot(df.year, df.totalprod)
plt.figure(figsize=(20,10))

chart = sns.boxplot(x='year', y='totalprod', data=df)
plt.figure(figsize=(10,4))

sns.pointplot(x='year', y='totalprod', data=df)
plt.figure(figsize=(10,4))

sns.barplot(x='year', y='totalprod', data=df)
g = sns.FacetGrid(df, col='state', col_wrap=10, size=3)

g.map(plt.plot, "year", "totalprod", marker="+")
df.describe().transpose()
#Total unique states

print(df.state.unique())

print("Total number of states = ", df.state.nunique())
## State with max production with year

df[df.totalprod == df.totalprod.max()][['state', 'year', 'totalprod']]
## average yield per col 

df.groupby('year').mean().round(3)[['yieldpercol']]
corr = df[['numcol', 'yieldpercol', 'totalprod', 'stocks', 'priceperlb', 'prodvalue']].corr()

corr
sns.heatmap(corr, annot=True, cmap='plasma', vmax=1, vmin=-1)