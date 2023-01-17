# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/womens-international-football-results/results.csv')
data.head()
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
plt.figure(figsize=(12, 6))



data.groupby('year')['date'].count().plot()



# World Cups

plt.axvline(x=1991, color='k', linestyle='--')

plt.axvline(x=1995, color='k', linestyle='--')

plt.axvline(x=1999, color='k', linestyle='--')

plt.axvline(x=2003, color='k', linestyle='--')

plt.axvline(x=2007, color='k', linestyle='--')

plt.axvline(x=2011, color='k', linestyle='--')

plt.axvline(x=2015, color='k', linestyle='--')

plt.axvline(x=2019, color='k', linestyle='--')
# First World Cup in 1991

# Pick in the Olympics in 2018

# COVID stopped the rise of matchs in women's football
for i, row in data.iterrows():

    if (row['home_score'] - row['away_score']) > 0:

        data.loc[i,'home_result'] = 'win'

        data.loc[i,'away_result'] = 'loss'

    elif (row['home_score'] - row['away_score']) < 0:

        data.loc[i,'home_result'] = 'loss'

        data.loc[i,'away_result'] = 'win'

    else:

        data.loc[i,'home_result'] = 'draw'

        data.loc[i,'away_result'] = 'draw'
home = data[['home_team', 'home_result']]

home.columns = ['team', 'result']



away = data[['away_team', 'away_result']]

away.columns = ['team', 'result']
res = pd.concat([home,away])
res = res[res.result == 'win'].groupby('team').count().reset_index()

res = res.sort_values('result', ascending=False)

res
fig, ax = plt.subplots(figsize=(8, 8))

ax.pie(res[:10].result, labels=res[:10].team, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')



plt.show()
data.tournament.unique()
wc = data[data.tournament=='FIFA World Cup']
wc = wc[['home_team', 'away_team', 'home_result', 'away_result', 'year']]
# get all finals

wc = wc.groupby('year').tail(1)
wc
# change results for the 2 draws

wc.loc[(wc['year']==1999), 'home_result'] = 'win'

wc.loc[(wc['year']==1999), 'away_result'] = 'loss'

wc.loc[(wc['year']==2011), 'home_result'] = 'win'

wc.loc[(wc['year']==2011), 'away_result'] = 'loss'
home = wc[['home_team', 'home_result']]

home.columns = ['team', 'result']



away = wc[['away_team', 'away_result']]

away.columns = ['team', 'result']
wc = pd.concat([home,away])
wc = wc[wc.result == 'win'].groupby('team').count().reset_index()

wc = wc.sort_values('result', ascending=False)
plt.figure(figsize=(12, 6))

sns.barplot(x="team", y="result", data=wc)
euro = data[data.tournament.isin(['Euro', 'UEFA Euro'])]
euro = euro[['home_team', 'away_team', 'home_result', 'away_result', 'year']]
# get all finals

euro = euro.groupby('year').tail(1)
euro
home = euro[['home_team', 'home_result']]

home.columns = ['team', 'result']



away = euro[['away_team', 'away_result']]

away.columns = ['team', 'result']
euro = pd.concat([home,away])
euro = euro[euro.result == 'win'].groupby('team').count().reset_index()

euro = euro.sort_values('result', ascending=False)
plt.figure(figsize=(12, 6))

sns.barplot(x="team", y="result", data=euro)
# A lot of domination from Germany