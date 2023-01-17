# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/Suicides Rates Overview 1985 to 2016/master.csv"))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
dataset = pd.read_csv('../input/master.csv')
dataset.head()
dataset.tail()
dataset.info()
dataset.describe()
dataset.shape
dataset.columns
del dataset['country-year']
del dataset['HDI for year']
dataset.rename(columns={'gdp_for_year ($) ' : 'gdp_for_year'},inplace = True)
dataset.rename(columns={'gdp_per_capita ($)':'gdp_per_capita'},inplace = True)
dataset.corr()
plt.figure(figsize= (10,7))

sns.heatmap(dataset.corr(),annot=True)
years = dataset.year.unique()

years = sorted(years)
population = []

for year in years:

    population.append([dataset[dataset['year']==year]['population'].sum()])



plt.plot(years, population,'-o')

plt.ylabel('Population -->')

plt.xlabel('Years --> ')

plt.show()
suicides = []

for year in years:

    suicides.append([dataset[dataset['year']==year]['suicides_no'].sum()])

plt.plot(years, suicides,'-o')

plt.ylabel('Suicides -->')

plt.xlabel('Years --> ')

plt.show()
plt.figure(figsize=(10,7))

sns.barplot(x='age',y='suicides_no',hue='sex',data=dataset)

plt.show()
generation = pd.unique(dataset['generation'])

gen_pos = np.arange(len(generation))



g_suic=[dataset[dataset['generation']==gen]['suicides_no'].sum() for gen in generation]

    

plt.barh(generation,g_suic)

plt.yticks(gen_pos,generation)
plt.figure(figsize=(10,25))

sns.countplot(y='country',data=dataset,alpha=0.7)