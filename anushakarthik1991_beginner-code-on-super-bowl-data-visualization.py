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
import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('/kaggle/input/superbowl-history-1967-2020/superbowl.csv')
data.head()
data.info()
data.Winner.mode()
data.Loser.mode()
pd.DataFrame(data.Winner.value_counts()).reset_index().rename(columns = { 'index' : 'Team'})
pd.DataFrame(data.Loser.value_counts()).reset_index().rename(columns = { 'index' : 'Team'})
pd.DataFrame(data.MVP.value_counts()).reset_index().rename(columns = { 'index' : 'Player'} )
round(data.describe())
print('The maximum points scored by winning team in the game is' +str(' ')+ str(data['Winner Pts'].max()))
print(' The maximum points scored by losing team in the game is' +str(' ')+str(data['Loser Pts'].max()))
data['score diff'] = data['Winner Pts'] - data['Loser Pts']
data['score diff']
import plotly.express as px

plt.figure(figsize=(40,18))

fig= px.scatter(data,x="Loser", y="Winner", size="score diff", hover_name= "Stadium", color="Date", title= 'super bowl 1967-2020 games resume')

fig.show()
svc=data.State.value_counts()

sbd=svc.reset_index()

sbd.rename(columns={'index': 'State', 'State': 'Number of played matches'}, inplace=True)

sbd
plt.figure(figsize=(50,18))

sns.countplot(x = 'State', data = data)
finals = list(set(list(data.Winner) + list(data.Loser)))

print('The teams that have made it to the SuperBowl are: \n' + str(finals) + str('\n\n\n') + str('Total count: ') + str(len(finals)))
finals_ap = list(data.Winner) + list(data.Loser)

max(set(finals_ap), key=finals_ap.count)

data['Times'] = 1

data.groupby('Winner')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)

data.groupby('Loser')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)
losers = data.groupby('Loser')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)

winners = data.groupby('Winner')['Times'].sum().reset_index(drop = False).sort_values(by = 'Times',ascending = False)

losers = losers.rename(columns = {'Loser' : 'Team'})

winners = winners.rename(columns = {'Winner' : 'Team'})
most_finals = losers.merge(winners, on = 'Team', how = 'left')
most_finals['Times_y'].fillna(0, inplace = True)

most_finals['most appeareances'] = (most_finals['Times_x'] + most_finals['Times_y']).astype(int)

most_finals.drop(['Times_x', 'Times_y'], axis = 1, inplace = True)

most_finals.sort_values(by = 'most appeareances', ascending = False)
plt.figure(figsize=(40,20))

sns.countplot(x='most appeareances',data=most_finals.sort_values(by = 'most appeareances', ascending = False),palette='coolwarm')