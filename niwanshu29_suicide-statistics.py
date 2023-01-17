# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import seaborn as sns
import numpy as np
sns.set(style = "ticks")
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib
from matplotlib.colors import LogNorm
from scipy import stats
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/who-suicide-statistics/who_suicide_statistics.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset.dropna(inplace = True)
dataset.shape
dataset.age.value_counts()
dataset.suicides_no.mean()
sns.set(rc = {'figure.figsize':(15,10)})
sns.regplot(x = dataset.year , y = dataset.suicides_no , x_jitter = 0.2 , order = 4)
plt.yscale('log')
plt.show()
dataset.groupby(['country','age']).suicides_no.sum().nlargest(10).plot(kind = 'barh')
sns.catplot(x = 'sex' , y = 'suicides_no',data = dataset,col = 'age',kind = 'bar' , estimator = np.median , height= 4)
plt.show()
dx = dataset.groupby(['age','sex']).agg({'suicides_no': np.sum}).unstack()
dx
pp =sns.catplot(x = 'age',y = 'suicides_no',hue = 'sex',col = 'year',data = dataset , col_wrap = 3 , estimator = sum,kind = 'bar')
plt.xticks(rotation = 90)
plt.show()
dx = dataset.groupby(['year','age']).agg({'suicides_no':np.sum}).reset_index()
sns.lineplot('year','suicides_no',hue='age',style='age',data=dx,palette="ch:2.5,.25",sort=False)
plt.show()
g = sns.FacetGrid(dataset , row = 'sex',col = 'age',margin_titles=True)
g.map(plt.scatter,"suicides_no",'population',edgecolor = 'w')
plt.show()
g = sns.FacetGrid(dataset,row = 'year',col = 'age',margin_titles= True)
g.map(plt.scatter , "suicides_no",'population',edgecolor = 'w')
plt.show()
g = sns.FacetGrid(dataset.groupby(['country','year']).suicides_no.sum().reset_index() , col = 'country',col_wrap = 3)
g.map(plt.plot , 'year','suicides_no',marker = '.')
plt.show()
p = pd.crosstab(dataset.country , dataset.year , values = dataset.suicides_no , aggfunc = 'sum')
sns.heatmap(p.loc[:,2011:2015].sort_values(2015,ascending = False).dropna().head(10),annot = True)
plt.show()
dataset.groupby('country')['suicides_no'].sum().reset_index().sort_values('suicides_no',ascending = True).tail(15).plot(x = 'country', y = 'suicides_no',kind = 'barh')
dataset.groupby('country')['population'].sum().reset_index().sort_values('population').tail(15).plot(kind = 'barh',y = 'population',x = 'country')
sns.jointplot(dataset.suicides_no , dataset.population , kind = 'scatter').annotate(stats.pearsonr)
plt.show()


