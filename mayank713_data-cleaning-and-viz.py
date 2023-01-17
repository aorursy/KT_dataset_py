# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv', encoding='ISO-8859-1')
data.head()
data.info()
def row(player):

    if '*' in player['HS']:

        return 1

    else:

        return 0



data['hs_not_out'] = data.apply(row, axis=1)
data.head()
data['HS'] = data['HS'].str.replace('*', '')

data['HS'] = data['HS'].str.replace('-', '0')

data['Mat'] = data['Mat'].replace('-', '0')

data['Inn'] = data['Inn'].str.replace('-', '0')

data['NO'] = data['NO'].str.replace('-', '0')

data['Runs'] = data['Runs'].str.replace('-', '0')

data['100'] = data['100'].str.replace('-', '0')

data['50'] = data['50'].str.replace('-', '0')

data['0'] = data['0'].str.replace('-', '0')

data['Avg'] = data['Avg'].str.replace('-', '0')
data['Inn'] = data['Inn'].astype('int32')

data['NO'] = data['NO'].astype('int32')

data['Runs'] = data['Runs'].astype('int32')

data['HS'] = data['HS'].astype('int32')

data['Avg'] = data['Avg'].astype('float32')

data['100'] = data['100'].astype('int32')

data['50'] = data['50'].astype('int32')

data['0'] = data['0'].astype('int32')
data.info()
# histogram for the avg. innings played

plt.figure(figsize=(15,6))

sns.distplot(data['Inn'])
plt.figure(figsize=(15,6))

sns.lineplot(data['Inn'], data['100'])

plt.title('Np. of innings played vs centuries scored')
# scatterplot for the no. of centuries scored for the innings played

sns.scatterplot(data['Inn'], data['100'])
# high score from a player

top_10_HS = data['HS'].sort_values(ascending=False)[:10].index

top_10 = data.iloc[top_10_HS]

top_10['HS'].plot.bar()

y = np.arange(10)

plt.xticks(y, labels=top_10['Player'])

plt.show()
plt.figure(figsize=(15,6))

sns.scatterplot(data['HS'], data['Inn'], hue=data['hs_not_out'])

plt.title('High score vs no. of innings played')
# distribution of whether batsman was out or not out when they scored their high score

f, ax = plt.subplots(1,2,figsize=(15,6))

data['hs_not_out'].value_counts().plot.bar(ax=ax[0])

labels = ['out', 'not out']

data['hs_not_out'].value_counts().plot.pie(ax=ax[1], autopct='%1.1f%%', labels=labels)
# no. of 100, 50 , 0 runs scored

x = [100, 50, 0]

y = [data['100'].sum(), data['50'].sum(), data['0'].sum()]

sns.barplot(x,y)
# top 10 avg runs by a player

plt.figure(figsize=(15,6))

top_10_avg = data['Avg'].sort_values(ascending=False)[:10].index

top_10 = data.iloc[top_10_avg]

top_10['Avg'].plot.bar()

y = np.arange(10)

plt.xticks(y, labels=top_10['Player'])

plt.show()
# most 100's by a player

plt.figure(figsize=(15,6))

top_100_i = data['100'].sort_values(ascending=False)[:10].index

top_100 = data.iloc[top_100_i]

top_100

top_100['100'].plot.bar()

y = np.arange(10)

plt.xticks(y, top_100['Player'])

plt.title('Top the 100 scored by a player')

plt.show()