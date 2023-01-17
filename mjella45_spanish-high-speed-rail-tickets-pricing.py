# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # used for plot interactive graph.



import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/renfe.csv', parse_dates=['insert_date', 'start_date', 'end_date'])
data.info()

data.head()
print("Percentage null or na values in df")

((data.isnull() | data.isna()).sum() * 100 / data.index.size).round(2)
data.price = data.price.replace("NEW", np.nan)

data.dropna(how ='any', inplace = True)

del data['Unnamed: 0']
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="destination",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Tickets to destination',size = 20)
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="origin",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Tickets sales from origin',size = 20)
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="train_type",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Train Type',size = 20)
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="train_class",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Train class',size = 20)
rcParams['figure.figsize'] = 15,7

g = sns.countplot(x="fare",data=data, palette = "Set1")

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")

g 

plt.title('Fare',size = 20)
X=data

X.head()

#X.price = X.price.apply(lambda x: x.replace(',',''))

X.price = X.price.astype(int)

fig, ax = plt.subplots(figsize=[16,9])

sns.distplot(X['price'],ax=ax,color='green')

ax.set_title('Price Distrubution for all tickets')
sns.violinplot(x=data.destination, y=data.price);

plt.title('Price Distrubution of tickets with respect citys')
sns.violinplot(x=data.train_type, y=data.price);

plt.title('Price Distrubution of tickets with respect to Train Type')
sns.violinplot(x=data.origin, y=data.price);

plt.title('Price Distrubution of tickets with respect to train origin')


sns.set(style="whitegrid")

sns.catplot(x="origin", y="price",  kind="bar",palette='coolwarm', data=data);

plt.title(' Average Ticket Price with respect to train origin')
sns.catplot(x="fare", y="price",kind="point", data=data);

data.groupby(['destination','origin']).size().plot(kind='bar', stacked=True)

plt.title('Ticket sales between madrid and other citys')

plt.ylabel('Number of tickets')

plt.xlabel('Citys')