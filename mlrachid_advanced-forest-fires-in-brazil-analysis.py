# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_palette('husl')

import missingno as msn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

data.head()

msn.matrix(data)
data.info()
data['month'].unique()
df = data.drop('date', axis=1, inplace=True)

df = data.rename({'year':'Year', 'state':'State', 'number':'Fires', 'month':'Month'}, axis=1)

df.head()
df['Month'].unique()
english_months = {'Janeiro':'January', 'Fevereiro':'February', 'Março':'March', 'Abril':'April', 'Maio':'May', 'Junho':'June', 'Julho':'Jully', 'Agosto':'August', 'Setembro':'September', 'Outubro':'October', 'Novembro':'November', 'Dezembro':'December'}

df['Month'] = df['Month'].map(english_months)

df['Month'].unique()
fires_per_year = df.groupby(['Year'], as_index=None)['Fires'].agg('sum').round(0)

fires_per_year.head()
fires_per_year['Fires'].describe()
fig, ax = plt.subplots(1,1, figsize=(12, 7), dpi=72)

sns.regplot(data=fires_per_year, x='Year', y='Fires', ax=ax)

plt.xticks(np.arange(1998, 2018, 1))

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(16, 6))

sns.distplot(fires_per_year['Fires'], ax=ax, color='purple')


fig, ax = plt.subplots(figsize=(12, 8))

sns.scatterplot(data=fires_per_year, x='Year', y='Fires', color='red', ax=ax)

sns.lineplot(data=fires_per_year, x='Year', y='Fires', color='red', ax=ax)

plt.xticks(np.arange(1998, 2018, 1))





months = list(df['Month'].unique())

months_fires = []

for i in list(np.arange(0, 12)):

    month_fire = df.query('Month in ["'+months[i]+'"]').groupby(['Year'], as_index=None)['Fires'].sum().round(0)

    month_fire = month_fire.rename({'Fires':months[i]}, axis=1)

    if i == 0:

        months_fires.append(month_fire)

    else:

        months_fires.append(pd.merge(months_fires[i-1], month_fire, on='Year'))

fires_per_month = months_fires[-1]

fires_per_month.head()


fires_per_month.plot.barh(x='Year', y=months[:6], figsize=(18, 28))
fires_per_month.plot.barh(x='Year', y=months[6:], figsize=(18, 28))
fig, ax = plt.subplots(1, 1)

corr = fires_per_month[months].corr()

sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',linewidths=.05)

fig.subplots_adjust(top=0.93)

fig.suptitle('Months correlation heatmap', fontsize=14)
states = list(df['State'].unique())

states_fires = []

for i in list(np.arange(0, len(states))):

    state_fire = df.query('State in ["'+states[i]+'"]').groupby(['Year'], as_index=None)['Fires'].sum().round(0)

    state_fire = state_fire.rename({'Fires':states[i]}, axis=1)

    if i == 0:

        states_fires.append(state_fire)

    else:

        states_fires.append(pd.merge(states_fires[i-1], state_fire, on='Year'))

fires_per_state = states_fires[-1]

fires_per_state.head()
fires_per_state[states].sum().sort_values(ascending=False)
hot_states = fires_per_state[list(fires_per_state.sum().nlargest(12).index)]

hot_states = pd.merge(fires_per_year, hot_states)

hot_states
interesting_states = list(hot_states.columns)

fig, ax = plt.subplots(1, 1, figsize=(18, 28))

hot_states.plot.barh(x='Year', y=interesting_states[2:], ax=ax)