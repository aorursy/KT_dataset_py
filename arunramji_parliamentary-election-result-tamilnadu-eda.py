# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
#reading files

A = pd.read_csv('../input/tnresult/tamilnadu_result.csv' )

B = pd.read_csv('../input/literacy/literacy_rate_constituency.csv')

A.head()
#basic summary

A.drop(['percent_votes','rank'],axis=1).describe()
#constitency wise basic summary stats

x = A.drop(['percent_votes','rank'],axis=1).groupby('Constituency').describe()

x.transpose()
plt.figure(figsize=(10,6))

A.hist(column='total_votes',bins=5);

plt.title('Distribution of totalvotes')

plt.show()
plt.figure(figsize=(13,7))

plt.bar(A['Constituency'],A['total_votes'],width=0.8)

plt.xticks(A['Constituency'], rotation='vertical')

plt.xlabel('Constituency')

plt.ylabel('total_votes')

plt.title('constituency wise total votes')

plt.show()

z = A.groupby('party').sum()
z = z.drop(['percent_votes','rank'],axis=1).sort_values('evm_votes',ascending=False)

z
sns.distplot(z['total_votes'])

plt.grid()

plt.show()


z.plot(y=['postal_votes','evm_votes'],kind='bar',figsize=(10,6),grid=True,title = 'party wise voteshare',width=1);

plt.show()
plt.figure(figsize=(10,7))

plt.grid()

plt.scatter(A['party'],A['total_votes'])

plt.xticks(A['party'],rotation='vertical')

plt.title('party wise total share')

plt.show()
p = A.pivot('Constituency','party','total_votes')

p
plt.figure(figsize=(10,6))

ax = sns.heatmap(p)

plt.title('party&constituency vote distribution')

sns.set(style="white")

plt.show()
plt.figure(figsize=(10,5))

sns.boxenplot(x=A.party, y=A.total_votes);

plt.grid()

plt.xticks(rotation='vertical')

plt.title('Vote distribution using box')

plt.show()
xx = A.groupby('Constituency').sum()

w = pd.merge(xx,B,how='inner',on='Constituency')

w.head()
plt.figure(figsize=(10,6))

plt.title('Literacy vs Voting')

sns.set(style='darkgrid')

sns.regplot(x="literacy", y="total_votes", data=w)

plt.show()
