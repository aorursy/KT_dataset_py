# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

import seaborn as sns

import matplotlib as ml

import matplotlib.pyplot as plt

%matplotlib inline

ml.style.use('ggplot')



init_notebook_mode(connected=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pbg = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

pbg.head()
pbg.tail(50)
pbg.info()
pbg.isnull().sum()
pbg.describe()
# creating a copy

pg = pbg.copy()
pg[pg.winPlacePerc.isnull()]
pg.dropna(axis=0,inplace = True)
pg.isnull().sum()
pbg = pg

pbg.isnull().sum()
plt.figure(figsize=(30,20))

sns.heatmap(pbg.corr(),annot=True)

plt.show()
pbg.head(20)
pbg.columns
# REMOVE HIGHLY CORRELATED FEATURES. HIGHLY CORRELATED TO 'KILLS'. POORLY CORRELATED TO 'WINPLACEPERC'.

pbg.drop(columns=['killPlace','headshotKills','killStreaks','longestKill'],axis=1,inplace=True)

pbg.info()
pbg.head(40)
wins_modr_best = []

for val in list(pbg.winPlacePerc.unique()):

    if val > 0.45:

        wins_modr_best.append(val)

    else:

        continue

print(pbg.winPlacePerc.nunique())

print(pd.Series(wins_modr_best).nunique())
winner = pbg[pbg.winPlacePerc==1]

loser = pbg[pbg.winPlacePerc==0]



fig = go.Figure(data=[go.Pie(labels=['Won','Lost','Drew/Others'],

                             values=[winner.shape[0],loser.shape[0],pbg.shape[0]-(winner.shape[0]+loser.shape[0])])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(line=dict(color='#000000', width=2)))

fig.show()
match_types = list(pbg.matchType.value_counts().values)

labels = list(pbg.matchType.value_counts().index)



# Plot a pie chart to show which game type is more popular

fig = go.Figure(data=[go.Pie(labels=labels, values=match_types, hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(line=dict(color='#000000', width=2)))

fig.show()
pw = pbg[pbg['winPlacePerc'] == 1]

pl = pbg[pbg['winPlacePerc'] == 0]
for_win = list(pw.matchType.value_counts().values)

for_loss = list(pl.matchType.value_counts().values)



fig = go.Figure(data=[

    go.Bar(name='WON', x=list(pw.matchType.values), y=for_win),

    go.Bar(name='LOST', x=list(pl.matchType.values), y=for_loss)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
plt.figure(figsize=(20,10))

sns.countplot(x='kills',data=pw)

plt.title("NO. OF MATCHES WON V/S NO. OF KILLS")

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(x='kills',data=pl)

plt.title("NO. OF MATCHES LOST V/S NO. OF KILLS")

plt.show()
sns.jointplot(x="winPlacePerc", y="kills", data=pbg, height=10, ratio=3, color="g")

plt.show()
plt.figure(figsize=(30,20))

sns.distplot(pw['damageDealt'])

sns.distplot(pl['damageDealt'])

plt.legend(['WON','LOST'])

plt.show()
sns.jointplot(x="winPlacePerc", y="damageDealt", data=pbg, height=10, ratio=3, color="r")

plt.show()
# Percentage of zero kills winners

colors1 = ['maroon','green']

colors2 = ['yellow']

fig = go.Figure(data=[go.Pie(labels=['ZERO KILLS','OTHERS'],

                             values=[pw[pw.kills==0].shape[0],(pw.shape[0]-pw[pw.kills==0].shape[0])])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors1,line=dict(color='#000000', width=2)))

fig.show()



# Percentage of zero damage winners

fig = go.Figure(data=[go.Pie(labels=['ZERO DAMAGE','OTHERS'],

                             values=[pw[pw.damageDealt==0].shape[0],(pw.shape[0]-pw[pw.damageDealt==0].shape[0])])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors2,line=dict(color='#000000', width=2)))

fig.show()
plt.figure(figsize=(30,20))

sns.distplot(pw['walkDistance'])

sns.distplot(pl['walkDistance'])

plt.legend(['WON','LOST'])

plt.show()
sns.jointplot(x='winPlacePerc', y='walkDistance', data=pbg, height=10, ratio=3, color="maroon")

plt.show()
plt.figure(figsize=(30,20))

sns.distplot(pw['rideDistance'])

plt.title('DISTRIBUTION OF RIDING DISTANCE OF WINNERS')

plt.show()
sns.jointplot(x='winPlacePerc', y='rideDistance', data=pbg, height=10, ratio=3, color="y")

plt.show()
plt.figure(figsize=(30,20))

sns.distplot(pw['swimDistance'],kde=False)

sns.distplot(pl['swimDistance'],kde=False)

plt.legend(['WON','LOST'])

plt.show()
sns.jointplot(x='winPlacePerc', y='swimDistance', data=pbg, height=10, ratio=3, color="g")

plt.show()
# Percentage of zero walk distance

colors1 = ['maroon','green']

colors2 = ['yellow']

colors3 = ['grey','red']

fig = go.Figure(data=[go.Pie(labels=['ZERO WALK DISTANCE','OTHERWISE'],

                             values=[pw[pw.walkDistance==0].shape[0],(pw.shape[0]-pw[pw.walkDistance==0].shape[0])])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors1,line=dict(color='#000000', width=2)))

fig.show()



# Percentage of zero riding distance

fig = go.Figure(data=[go.Pie(labels=['ZERO RIDE DISTNACE','OTHERWISE'],

                             values=[pw[pw.rideDistance==0].shape[0],(pw.shape[0]-pw[pw.rideDistance==0].shape[0])])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors2,line=dict(color='#000000', width=2)))

fig.show()



# Percentage of zero swimming distance

fig = go.Figure(data=[go.Pie(labels=['ZERO SWIM','OTHERWISE'],

                             values=[pw[pw.swimDistance==0].shape[0],(pw.shape[0]-pw[pw.swimDistance==0].shape[0])])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors3,line=dict(color='#000000', width=2)))

fig.show()
plt.figure(figsize=(30,20))

sns.pointplot(x='heals',y='winPlacePerc',data=pbg,color='red',alpha=0.8)

sns.pointplot(x='boosts',y='winPlacePerc',data=pbg,color='blue',alpha=0.8)

plt.legend(['BOOSTS'])

plt.xlabel('NUMBER OF HEALING/BOOSTING ITEMS USED')

plt.ylabel('Win Percentage')

plt.title('HEALS V/S BOOSTS')

plt.grid()

plt.show()
sns.jointplot(x='winPlacePerc', y='heals', data=pbg, height=10, ratio=3, color="maroon")

plt.show()



sns.jointplot(x='winPlacePerc', y='boosts', data=pbg, height=10, ratio=3, color="y")

plt.show()
plt.figure(figsize=(30,20))

sns.pointplot(x='weaponsAcquired',y='winPlacePerc',data=pbg,color='red',alpha=0.8)

plt.xlabel('NUMBER OF  WEAPONS ACQUIRED')

plt.ylabel('Win Percentage')

plt.title('Weapons Acquired')

plt.grid()

plt.show()
# WEAPONS ACQUIRED

sns.jointplot(x='winPlacePerc', y='weaponsAcquired', data=pbg, height=10, ratio=3, color="b")

plt.show()



# DBNOs

sns.jointplot(x='winPlacePerc', y='DBNOs', data=pbg, height=10, ratio=3, color="violet")

plt.show()