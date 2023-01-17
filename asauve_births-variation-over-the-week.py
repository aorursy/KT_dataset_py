import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from __future__ import print_function

sns.set()

df = pd.read_csv('../input/births.csv')

    

df.head()
df.describe()
df.day.unique()
df = df[(df.day>=1) & (df.day<=31)]

ndays = df[['year', 'day']].groupby('year').count()

plt.bar(ndays.index, ndays.day)

plt.title("Nb of days available per year (both genders)")

lines = plt.plot(plt.gca().get_xlim(), 365*2*np.ones(2), 'g-')
daycount = df.copy()

daycount.day.values[:] = 1

daycount.pivot_table('day', index=['year', 'gender'], columns=['month'], aggfunc=np.sum).T
ax = sns.distplot(df.births, norm_hist=False)
baddays = df[df.births < 1000].copy()[['year', 'month', 'day']]

baddays.groupby(['year', 'month']).count().T
df = df[df.births > 1000]
def showbirth():

    fig = plt.figure(figsize=(15,4))

    males = df.gender == 'M'

    females = df.gender == 'F'

    plt.plot(df.year[males]+df.month[males]/12.+df.day[males]/365., 

             df.births[males], 

             '+', label='Males')

    plt.plot(df.year[females]+df.month[females]/12.+df.day[females]/365., 

             df.births[females], 

             'x', label='Females')

    plt.xlabel('year')

    plt.ylabel('births')

    plt.legend()

    return fig

fig = showbirth()

plt.title('Nombre de naissance sur les jours entre 1969 et 1989')

limits = plt.xlim(1969, 1989)
import datetime

fig = showbirth()

plt.title('All days of 1970')

plt.xlim(1970, 1971)

xticks = plt.xticks(1970 + np.arange(12)/12.)

monthes = [ datetime.date(2000, month, 1).strftime('%B') for month in range(1, 13)]

labels = fig.get_axes()[0].set_xticklabels( monthes )
datetimes = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')



bad_dates = datetimes.isna()

print("Number of invalid date values found:", bad_dates.sum())

dayname = datetimes.map(lambda dt: dt.day_name())

df.insert(0, 'dayname', dayname)

df.head()

df['decade'] = pd.cut(df.year, [1960, 1970, 1980, 1990], labels=[ "60's", "70's", "80's "])

pivot = pd.pivot_table(df, values='births', index='dayname', columns='decade', aggfunc=np.mean)

plt.rcParams['figure.figsize'] = (10,5)

ax = pivot.plot()



# for an unknown reason daynames does not appear on this very plot

ax.set_xticks(np.arange(7)) 

ax.set_xticklabels(pivot.index)

pivot