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
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.shape
df.head(10)
checkdup = df.groupby(['Country/Region','Province/State', 'ObservationDate']).count().iloc[:,0]

checkdup[checkdup>1]
latest = df[df.ObservationDate == df.ObservationDate.max()]

print ('Total confirmed cases: %.d' %np.sum(latest['Confirmed']))

print ('Total death cases: %.d' %np.sum(latest['Deaths']))

print ('Total recovered cases: %.d' %np.sum(latest['Recovered']))

print ('Death rate %%: %.2f' % (np.sum(latest['Deaths'])/np.sum(latest['Confirmed'])*100))
dict(df['Country/Region'].value_counts())
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime



def plot_new(column, title):

    _ = cty.sort_values(column, ascending=False).head(20)

    g = sns.barplot(_[column], _.index)

    plt.title(title, fontsize=14)

    plt.ylabel(None)

    plt.xlabel(None)

    plt.grid(axis='x')

    for i, v in enumerate(_[column]):

        if column == 'Death Rate':

            g.text(v*1.01, i+0.1, str(round(v,2)))

        else:

            g.text(v*1.01, i+0.1, str(int(v)))



plt.figure(figsize=(9,16))

plt.subplot(311)

plot_new('Confirmed','Confirmed cases top 20 countries')

plt.subplot(312)

plot_new('Deaths','Death cases top 20 countries')

plt.subplot(313)

plot_new('Active','Active cases top 20 countries')



plt.show()
import matplotlib.dates as mdates

months_fmt = mdates.DateFormatter('%b-%e')



def plot_cty(num, evo_col, title):

    ax[num].plot(evo_col, lw=3)

    ax[num].set_title(title)

    ax[num].xaxis.set_major_locator(plt.MaxNLocator(7))

    ax[num].xaxis.set_major_formatter(months_fmt)

    ax[num].grid(True)

    



def evo_cty(country):

    evo_cty = df[df.Country==country].groupby('ObservationDate')[['Confirmed','Deaths','Recovered']].sum()

    evo_cty['Active'] = evo_cty['Confirmed'] - evo_cty['Deaths'] - evo_cty['Recovered']

    evo_cty['Death Rate'] = evo_cty['Deaths'] / evo_cty['Confirmed'] * 100

    plot_cty((0,0), evo_cty['Confirmed'], 'Confirmed cases')

    plot_cty((0,1), evo_cty['Deaths'], 'Death cases')

    plot_cty((1,0), evo_cty['Active'], 'Active cases')

    plot_cty((1,1), evo_cty['Death Rate'], 'Death rate')

    fig.suptitle(country, fontsize=16)

    plt.show()
def evo_cty(country):fig, ax = plt.subplots(2, 2, figsize=(12,9))

evo_cty('Italy')
def evo_cty(country):fig ,ax= plt.subplots (2, 2, figsize=(12, 9))

evo_cty('US')