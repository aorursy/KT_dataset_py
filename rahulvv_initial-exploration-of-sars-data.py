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

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from itertools import cycle

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
sars = pd.read_csv('/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv', parse_dates=['Date'])

sars.sort_values('Date', inplace=True)

sars = sars.set_index('Date')

sars.head()
sars.info()
grouped = sars.groupby(['Country']).max().sort_values(['Cumulative number of case(s)'], ascending=False).reset_index()

grouped.head()
affected= sars.groupby(['Country']).max()['Cumulative number of case(s)'].head(50).sort_values(ascending=True)

deaths = sars.groupby(['Country']).max()['Number of deaths'].head(50).reindex(affected.index, axis=0)

recovered = sars.groupby(['Country']).max()['Number recovered'].head(50).reindex(affected.index, axis=0)

pos = list(range(len(affected)))

fig, ax = plt.subplots(figsize=(10,10))

w = .3

plt.barh([p+(2*w) for p in pos], deaths, w, color=next(color_cycle), label='Deaths')

plt.barh([p+w for p in pos], recovered, w, color=next(color_cycle), label='Recovered')

plt.barh([p for p in pos], affected, w, color=next(color_cycle), label='Affected')



plt.ylim(min(pos)-w, max(pos)+w*4)

ax.set_yticks([p+w for p in pos])

ax.set_yticklabels(affected.index)

plt.legend()

plt.show()
fig, ax = plt.subplots(figsize=(10,10))

sns.barplot(y='Country', x='Cumulative number of case(s)', data=grouped.head(15), color=next(color_cycle),alpha=0.5, label='affected')

sns.barplot(y='Country', x='Number recovered', data=grouped.head(15), color=next(color_cycle),alpha=0.5, label='recovered')

sns.barplot(y='Country', x='Number of deaths', data=grouped.head(15), color=next(color_cycle),alpha=0.5, label='deaths')

plt.legend()

plt.show()
sns.set(rc={'figure.figsize':(12,12)})

sns.set(style='white')

for country in  sars.Country.unique()[0:15]:

    df = sars[sars.Country==country]['Cumulative number of case(s)']

    df.plot(label=str(country))

    plt.legend()
sns.set(rc={'figure.figsize':(12,12)})

sns.set(style='white')

for country in  sars.Country.unique()[0:15]:

    df = sars[sars.Country==country]['Number recovered']

    df.plot(label=str(country))

    plt.legend()
sns.set(rc={'figure.figsize':(12,12)})

sns.set(style='white')

for country in  sars.Country.unique()[0:15]:

    df = sars[sars.Country==country]['Number of deaths']

    df.plot(label=str(country))

    plt.legend()