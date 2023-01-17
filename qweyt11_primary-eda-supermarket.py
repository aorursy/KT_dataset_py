# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from collections import Counter

import itertools



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/50_SupermarketBranches.csv')



print(df.shape)

df.head()
for c in ('Advertisement Spend', 'Promotion Spend', 'Administration Spend', 'Profit'):

    df[c] = df[c] / 1000
plt.figure(figsize=(12, 8))



plt.scatter(df.Profit.values, df['Administration Spend'].values, c='blue', label='Administration')

plt.scatter(df.Profit.values, df['Promotion Spend'].values, c='red', label='Promotion', marker='P', s=64)

plt.scatter(df.Profit.values, df['Advertisement Spend'].values, c='green', label='Advertisement', marker='d', s=64)



plt.xlabel('$Profit$', fontsize=14)

plt.ylabel('$k\$$', fontsize=14)



plt.legend(fontsize=14)
sns.heatmap(df[['Advertisement Spend', 'Promotion Spend', 'Administration Spend', 'Profit']].corr(),

            annot=True)
df.groupby('State').corr()[['Profit']].unstack(level=1)['Profit'][['Advertisement Spend', 'Promotion Spend', 'Administration Spend']].plot(kind='bar', title ="correlations of Profit with Spends by States", figsize=(15, 10), legend=True, fontsize=12)
df2 = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/Ads_CTR_Optimisation.csv')



print(df2.shape)

df2.head()
_, ax = plt.subplots(1, 15, figsize=(15, 10), sharey=True)



for i in range(15):

    ax[i].spy(df2.values[i * 100:(i + 1) * 100, :])
u, s, vh = np.linalg.svd(df2.values)

plt.plot(s)
plt.bar(range(1, 11), df2.sum(axis=0))



plt.xticks(range(1, 11))

plt.xlabel('Ad', fontsize=14)

plt.ylabel('count', fontsize=14)
hs, xs = np.histogram(df2.sum(axis=1), bins=np.arange(7) - 0.5)

plt.bar(xs[:-1] + 0.5, hs)



plt.xlabel('number of clicks', fontsize=14)

plt.ylabel('count', fontsize=14)
with open('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/Market_Basket_Optimisation.csv', 'r') as fr:

    baskets = [l.strip().split(',') for l in fr.readlines()]
plt.hist([len(b) for b in baskets], bins=np.arange(20) - 0.5, label='basket size')



plt.legend(fontsize=12)

plt.xticks(np.arange(20))

plt.xlim(0.5, 17);
Counter([i for b in baskets for i in b]).most_common(10)
Counter(['| '.join(c) for b in baskets for c in itertools.combinations(b, 2)]).most_common(10)
df3 = pd.read_csv('/kaggle/input/marketing-data-for-a-supermarket-in-united-states/supermarket_marketing/Supermarket_CustomerMembers.csv', index_col='CustomerID')



print(df3.shape)

df3.head()
plt.figure(figsize=(10, 7))

plt.scatter(df3['Annual Income (k$)'].values, df3['Spending Score (1-100)'])



plt.xlabel('Annual Income', fontsize=14)

plt.ylabel('Spending Score', fontsize=14)