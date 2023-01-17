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
import seaborn as sns

import matplotlib.pyplot as plt



covid_us = pd.read_csv('/kaggle/input/covid19-in-usa/us_states_covid19_daily.csv')

covid_us.head()

covid_us["date"] = pd.to_datetime(covid_us["date"], format="%Y%m%d").dt.date.astype(str)



covid_us_grpstates = covid_us.groupby(['state'])['total','positive','negative','death'].sum()

#covid_us_grpstates.to_frame()

covid_us_grpstates = covid_us_grpstates.reset_index()

covid_us_grpstates = covid_us_grpstates.sort_values('total', ascending = False)

covid_us_grpstates_top15 = covid_us_grpstates[:15]

covid_us_grpstates_top15['Death_rate'] = covid_us_grpstates_top15['death']/covid_us_grpstates_top15['total']



import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style("darkgrid", {"axes.facecolor": ".9"})

fig, ax1 = plt.subplots(figsize=(10,5))

color = 'tab:blue'

ax1.set_title('US : Confirmed cased and death rate', fontsize=16)

ax1 = sns.barplot(x=covid_us_grpstates_top15['state'], y=covid_us_grpstates_top15['positive'], palette='BuGn_r',label = 'Confirmed cases',ax=ax1)

ax1.tick_params(axis='y')

ax1.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax1.legend(loc = 'upper left', shadow=True)

ax1.grid(False)



ax2 = ax1.twinx()

color = 'tab:red'

ax2 = sns.lineplot(x=covid_us_grpstates_top15['state'], y=covid_us_grpstates_top15['Death_rate'] ,color=color,palette='BuGn_r', label = 'Death_rate',ax=ax2)

ax2.set_xticklabels(ax1.get_xticklabels(),rotation=90)

ax2.tick_params(axis='y', color=color)

ax2.grid(False)

top_states = ['NY','WA','CA','FL','IL','TX','MA','MN','NC','WQ','CO']

covid_us_grpstates_top = covid_us[covid_us['state'].isin(top_states)]

covid_us_grpstates_top = covid_us_grpstates_top.groupby(['date','state'])['total','positive','negative','death'].sum().sort_values('positive',ascending = False)

covid_us_grpstates_top = covid_us_grpstates_top.reset_index()



df = covid_us_grpstates_top.pivot(index='date', columns='state', values='positive')

df.plot(figsize=(15,5),title = 'Daily confirmed cases', fontsize=10)



g = sns.FacetGrid(covid_us_grpstates_top, col = 'state', col_wrap=5, height=3, )

g.map(sns.lineplot, "date", "positive")

g.set_xlabel = ('')
df = covid_us_grpstates_top.pivot(index='date', columns='state', values='death')

df.plot(figsize=(15,5),title = 'Daily Deaths', fontsize=10)



g = sns.FacetGrid(covid_us_grpstates_top, col = 'state', col_wrap=5, height=3, )

g.map(sns.lineplot, "date", "death")

g.set_xlabel = ('')