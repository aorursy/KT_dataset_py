# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import rcParams

# Standard plotly imports

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/zomato.csv')

data.head()

data.shape

#data.tail()
data.columns
#data[['phone','name']][10:25]

#data.loc[data['rest_type']=='Bar']
data.describe()
data.columns
data.drop(['url'], inplace=True, axis=1)
data['rate'].unique()
data.head()
data.loc[data['votes'] > 15000]
data['dish_liked'].unique()
data['approx_cost(for two people)'].describe()
((data.isnull()|data.isna()).sum() *100/data.index.size).round(2)
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
data.head()
data.drop(['address', 'phone','location'],inplace=True, axis=1)
data.head()
data.rename(columns ={'approx_cost(for two people)': 'average_cost', 'listed_in(city)': 'locality','listed_in(type)': 'restaurant_type'}, inplace = True)
data.head()
data.rate = data.rate.replace("NEW", np.nan)

#Drop rows where any cell in that row is NA

data.dropna(how ='any', inplace = True)
data.shape
X = data

X.rate = X.rate.astype(str)

X.rate = X.rate.apply(lambda x: x.replace('/5',''))

X.rate = X.rate.apply(lambda x: float(x))

X.head()
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="locality",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('locality',size = 20)
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="rest_type",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Type of Restaurant',size = 20)
plt.rcParams['figure.figsize'] = (3, 4)

plt.style.use('_classic_test')



X['online_order'].value_counts().plot.bar(color = 'cyan')

plt.title('Online orders', fontsize = 20)

plt.ylabel('Number of orders', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

x = pd.crosstab(X['rate'], X['online_order'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('online order vs rate', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
data['online_order'].iplot(kind='hist', xTitle='online Order',

                  yTitle='count', title='')
X.head()

X.average_cost = X.average_cost.apply(lambda x: x.replace(',',''))

X.average_cost = X.average_cost.astype(int)

fig, ax = plt.subplots(figsize=[16,4])

sns.distplot(X['average_cost'],ax=ax)

ax.set_title('Cost Distribution for all restaurants')