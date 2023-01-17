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
data = pd.read_csv('/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv')
data.head()
data.columns
plt.figure(figsize=(20, 6))

sns.countplot(data.League, order = data['League'].value_counts().index)

l=plt.xticks(rotation=90)
data.Season.unique()
nba = data[data.League == 'NBA']
nba.groupby('nationality').count().reset_index().iloc[:,:2].sort_values('League', ascending=False).head(10)
# Not USA

plt.figure(figsize=(15, 6))

sns.countplot(nba[nba['nationality'] != 'United States'].nationality, order = nba[nba['nationality'] != 'United States']['nationality'].value_counts().iloc[:10].index)

l=plt.xticks(rotation=90)
# Not USA

plt.figure(figsize=(15, 6))

sns.countplot(nba[(nba['nationality'] != 'United States') & (nba['Season'] == '2019 - 2020')].nationality, order = nba[(nba['nationality'] != 'United States') & (nba['Season'] == '2019 - 2020')]['nationality'].value_counts().iloc[:10].index)

l=plt.xticks(rotation=90)
nba['Age'] = nba['Season'].str.split('-').str[1].astype(float) - nba['birth_year']
nba.groupby('Season')['Age'].mean()
hw = nba[['Season', 'height_cm', 'weight_kg',]]
hw = hw.groupby('Season').mean().reset_index()
hw
plt.figure(figsize=(15, 6))

ax = sns.lineplot(x="Season", y="height_cm", data=hw)

ax2 = ax.twinx()

ax2 = sns.lineplot(x="Season", y="weight_kg", data=hw, c='red')
# Height decreases a very little but weight decreases a lot (-+3kg) in 10 years
scorers = nba[['Season', 'Stage', 'Player', 'Team', 'GP', 'PTS']]
scorers['PTS/G'] = round(scorers['PTS'] / scorers['GP'],2)



# Regular season

scorers = scorers[scorers['Stage'] == 'Regular_Season']
idx = scorers.groupby('Season')['PTS/G'].transform(max) == scorers['PTS/G']

best_scorers = scorers[['Season', 'Player', 'Team', 'PTS/G']][idx]

best_scorers
plt.figure(figsize=(18, 6))

ax = sns.lineplot(x="Season", y="PTS/G", data=best_scorers)

ax.set(ylim=(27, 37))

for index, row in best_scorers.iterrows():

    ax.annotate(row['Player'], (row['Season'], row['PTS/G'] + 0.5),  xycoords='data', xytext=(-27, -10), textcoords='offset points', size=12)
pas = nba[['Season', 'Stage', 'Player', 'Team', 'GP', 'AST']]
pas['AST/G'] = round(pas['AST'] / pas['GP'],2)



# Regular season

pas = pas[pas['Stage'] == 'Regular_Season']
idx = pas.groupby('Season')['AST/G'].transform(max) == pas['AST/G']

best_pas = pas[['Season', 'Player', 'Team', 'AST/G']][idx]

best_pas
plt.figure(figsize=(18, 6))

ax = sns.lineplot(x="Season", y="AST/G", data=best_pas)

ax.set(ylim=(9.5, 12))

for index, row in best_pas.iterrows():

    ax.annotate(row['Player'], (row['Season'], row['AST/G']),  xycoords='data', xytext=(-27, -10), textcoords='offset points', size=12)
reb = nba[['Season', 'Stage', 'Player', 'Team', 'GP', 'REB']]
reb['REB/G'] = round(reb['REB'] / reb['GP'],2)



# Regular season

reb = reb[reb['Stage'] == 'Regular_Season']
idx = reb.groupby('Season')['REB/G'].transform(max) == reb['REB/G']

best_reb = reb[['Season', 'Player', 'Team', 'REB/G']][idx]

best_reb
plt.figure(figsize=(18, 6))

ax = sns.lineplot(x="Season", y="REB/G", data=best_reb)

ax.set(ylim=(12, 16.5))

for index, row in best_reb.iterrows():

    ax.annotate(row['Player'], (row['Season'], row['REB/G']),  xycoords='data', xytext=(-27, -10), textcoords='offset points', size=12)
stl = nba[['Season', 'Stage', 'Player', 'Team', 'GP', 'STL']]
stl['STL/G'] = round(stl['STL'] / stl['GP'],2)



# Regular season

stl = stl[stl['Stage'] == 'Regular_Season']
idx = stl.groupby('Season')['STL/G'].transform(max) == stl['STL/G']

best_stl = stl[['Season', 'Player', 'Team', 'STL/G']][idx]

best_stl
plt.figure(figsize=(18, 6))

ax = sns.lineplot(x="Season", y="STL/G", data=best_stl)

ax.set(ylim=(2, 2.6))

for index, row in best_stl.iterrows():

    ax.annotate(row['Player'], (row['Season'], row['STL/G']),  xycoords='data', xytext=(-27, -10), textcoords='offset points', size=12)
blk = nba[['Season', 'Stage', 'Player', 'Team', 'GP', 'BLK']]
blk['BLK/G'] = round(blk['BLK'] / blk['GP'],2)



# Regular season

blk = blk[blk['Stage'] == 'Regular_Season']
idx = blk.groupby('Season')['BLK/G'].transform(max) == blk['BLK/G']

best_blk = blk[['Season', 'Player', 'Team', 'BLK/G']][idx]

best_blk
plt.figure(figsize=(18, 6))

ax = sns.lineplot(x="Season", y="BLK/G", data=best_blk)

ax.set(ylim=(2.3, 3.8))

for index, row in best_blk.iterrows():

    ax.annotate(row['Player'], (row['Season'], row['BLK/G']),  xycoords='data', xytext=(-27, -10), textcoords='offset points', size=12)
pm3 = nba[['Season', 'Stage', 'Player', 'Team', '3PM', '3PA']]
pm3['3P%'] = round(round(pm3['3PM'] / pm3['3PA'],3)*100,3)



# Regular season

pm3 = pm3[pm3['Stage'] == 'Regular_Season']
idx = pm3.groupby('Season')['3PM'].transform(max) == pm3['3PM']

best_pm3 = pm3[['Season', 'Player', 'Team', '3PM', '3P%']][idx]

best_pm3
plt.figure(figsize=(18, 6))

ax = sns.lineplot(x="Season", y="3PM", data=best_pm3)

ax.set(ylim=(150, 420))

for index, row in best_pm3.iterrows():

    ax.annotate(row['Player'] +' ' + str(row['3P%'])+'%', (row['Season'], row['3PM']+10),  xycoords='data', xytext=(-27, -10), textcoords='offset points', size=12)
plt.figure(figsize=(18, 6))

ax = sns.barplot(x="Season", y="3PM", data=best_pm3)

ax2 = ax.twinx()

ax2 = sns.lineplot(x="Season", y="3P%", data=best_pm3)

for index, row in best_pm3.iterrows():

    ax2.annotate(row['Player'].split( )[1] +' ' + str(row['3P%'])+'%', (row['Season'], row['3P%']),  xycoords='data', xytext=(-27, -10), textcoords='offset points', size=12)