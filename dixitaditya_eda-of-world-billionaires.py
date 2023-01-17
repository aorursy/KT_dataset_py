import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
base = pd.read_csv('../input/billionaires.csv')

base['net_worth'] = base.net_worth.astype(float)

base.head()
base.describe(include='all')

base.drop('rank',axis=1).hist()

#Nationality wise distribution

base.natinality.value_counts()
filter = ['Bill Gates','Jeff Bezos','Mark Zuckerberg','Paul Allen','Larry Ellison']

comparison = base[base['name'].isin(filter)][['name','year','net_worth']]

%matplotlib inline

sns.lineplot(data=comparison,x='year',y='net_worth',hue='name')



plt.xlabel('Year')

plt.ylabel('Net Worth')

plt.title('Wealth of Major Tech Companies Owners over years')

plt.show()
base.source_wealth.value_counts()

#Doing basic data cleaning of wealth source

dict = {'Wal-Mart':'Walmart','LVMH Moët Hennessy • Louis Vuitton':'LVHM'}

base.source_wealth.replace(dict,inplace=True)
top5 = base.groupby('source_wealth')['year'].nunique().sort_values(ascending=False).head().index

t = base[base['source_wealth'].isin(top5)]



sns.lineplot(data=t,x='year',y='net_worth',hue='source_wealth')

plt.xlabel('Year')

plt.ylabel('Net Worth')

plt.title('Wealth created by top5 most consistently appearning companies')

plt.show()
