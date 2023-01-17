# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from matplotlib.pyplot import xticks

from scipy.stats import skew

import seaborn as sns

import os

import warnings

import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
metadata = pd.read_csv('../input/Metadata.txt', sep='\t')
metadata
data = pd.read_csv('../input/Data.csv', encoding = 'ISO-8859-1')
data.head()
data.Series.unique()
top3 = data[(data.Winner.isin(['Federer R.', 'Nadal R.', 'Djokovic N.'])) | (data.Loser.isin(['Federer R.', 'Nadal R.', 'Djokovic N.']))]

top3 = top3[['Date', 'Winner', 'Loser', 'WRank', 'LRank']]
top3_w = top3[top3.Winner.isin(['Federer R.', 'Nadal R.', 'Djokovic N.'])]

top3_l = top3[top3.Loser.isin(['Federer R.', 'Nadal R.', 'Djokovic N.'])]



top3_w = top3_w[['Date', 'Winner', 'WRank']]

top3_l = top3_l[['Date', 'Loser', 'LRank']]
top3_w.columns = ['Date','Player','Rank']

top3_l.columns = ['Date','Player', 'Rank']
top3 = pd.concat([top3_w, top3_l], sort=False)

top3['Date'] = pd.to_datetime(top3.Date, format='%d/%m/%Y')

top3 = top3.sort_values(['Date'])
top3.Rank = top3.Rank.astype(int)



# Remove outline Ranks

top3 = top3[top3.Rank < 100]
federer = top3[top3.Player == 'Federer R.']

nadal = top3[top3.Player == 'Nadal R.']

djokovic = top3[top3.Player == 'Djokovic N.']
fig = plt.figure(figsize=(12,8))

sns.lineplot(x='Date', y='Rank', data=federer)

sns.lineplot(x='Date', y='Rank', data=nadal)

sns.lineplot(x='Date', y='Rank', data=djokovic)

fig.legend(bbox_to_anchor=(-0.2, 0.8, 1., 0), labels=['Federer','Nadal','Djokovic'])

t=fig.suptitle('Rank Evolution for the Big 3')
# Let's zoom on the top 10 

fig = plt.figure(figsize=(18,10))

sns.lineplot(x='Date', y='Rank', data=federer[federer.Rank <=10])

sns.lineplot(x='Date', y='Rank', data=nadal[nadal.Rank <=10])

sns.lineplot(x='Date', y='Rank', data=djokovic[djokovic.Rank <=10])

fig.legend(bbox_to_anchor=(-0.2, 0.8, 1., 0), labels=['Federer','Nadal','Djokovic'])

t=fig.suptitle('Rank Evolution for the Big 3 in the top 10')
# 2005 : First French Open for Nadal (N°3)

# 2004 : Federer won Australian Open and became N°1

# 2009 : Nadal injured, lost early in French Open and Wimbledon

# 2011 : Djokovic won 3 Grand Slams

# 2014 : Federer injured

# 2013 and 2016 : Nadal injured
# Grand Slams titles for the Big 3
slams = data[['Date','Tournament','Series', 'Round', 'Winner']]
slams = slams[(slams.Series == 'Grand Slam') & (slams.Round == 'The Final')]
slams = slams[slams.Winner.isin(['Federer R.', 'Nadal R.', 'Djokovic N.'])]
slams.head()
slams['Titles'] = slams.groupby('Winner').cumcount().astype(int) + 1

slams['Date'] = pd.to_datetime(slams.Date, format='%d/%m/%Y')

slams = slams.sort_values(['Date'])

slams.head()
federer_slams = slams[slams.Winner == 'Federer R.']

nadal_slams = slams[slams.Winner == 'Nadal R.']

djokovic_slams = slams[slams.Winner == 'Djokovic N.']
fig = plt.figure(figsize=(15,8))

sns.lineplot(x='Date', y='Titles', data=federer_slams)

sns.lineplot(x='Date', y='Titles', data=nadal_slams)

sns.lineplot(x='Date', y='Titles', data=djokovic_slams)

fig.legend(bbox_to_anchor=(-0.2, 0.8, 1., 0), labels=['Federer','Nadal','Djokovic'])

t=fig.suptitle('Slams Evolution for the Big 3')
# Grand Slams Wins per Rank
slams_winners = data[['Series', 'Round', 'WRank']]
slams_winners = slams_winners[(slams_winners.Series == 'Grand Slam') & (slams_winners.Round == 'The Final')]
slams_winners.WRank = slams_winners.WRank.astype(int)
slams_winners
fig = plt.figure(figsize=(15,8))

sns.distplot(slams_winners.WRank)
# Very rare after the 20th place

fig = plt.figure(figsize=(15,8))

sns.distplot(slams_winners.WRank[slams_winners.WRank <=10])
rank_prob_win = round(slams_winners.groupby('WRank')['Series'].count()/len(slams_winners),4)*100
top3_prob_win = rank_prob_win[1] + rank_prob_win[2] + rank_prob_win[3]

print("You have %s%% chances to win a Grand Slam if you are in the top 3" %top3_prob_win)
num1 = data[['Winner', 'Loser', 'WRank', 'LRank']]

num1 = num1[(num1.WRank != 'NR') & (num1.LRank != 'NR')]

num1 = num1.dropna()

num1['WRank'] = num1['WRank'].astype(int)

num1['LRank'] = num1['LRank'].astype(int)

num1 = num1[(num1.WRank == 1) | (num1.LRank == 1)]
num1_w = num1[num1.WRank == 1]['Winner']

num1_l = num1[num1.LRank == 1]['Loser']

num1_w.columns = ['Player']

num1_l.columns = ['Player']

num1 = pd.concat([num1_w, num1_l], sort=False)



num1 = num1.drop_duplicates()

print('Since 2000, there were %s Number 1 players' % num1.count())
# Upset in Grand Slam
slams = data[data.Series == 'Grand Slam']
slams.head()
upset = slams[['Tournament', 'Series', 'Round', 'AvgW', 'AvgL']]
upset = upset.dropna()
upset.head()
round(upset.groupby('Tournament')['AvgW'].mean(),3)
# Wimbledon is the Grand Slam with the best upsets
round(upset.groupby('Round')['AvgW'].mean(),3)
# There are more upset in Finals
# Set losed per Grand Slams
sets = slams[['Tournament', 'Series', 'Round', 'Wsets', 'Lsets']]

sets = sets.dropna()
round(sets.groupby('Tournament')['Lsets'].mean(),3)
# French Open is the tournament where player lose the less sets
round(sets.groupby(['Round'])['Lsets'].mean(),3)
# As expected, players lose more sets when they advanced in the tournament (the players have a better level)
# Grand Slam wins during period
wins = slams[['Tournament', 'Round', 'Winner']]

wins = wins[wins.Round == 'The Final']
wins.head()
winners = wins.groupby('Winner')['Tournament'].count()
winners = winners.reset_index()
winners = winners.sort_values(['Tournament'], ascending=False)

winners
plt.figure(figsize=(15,8))



g = sns.barplot(x=winners.Winner, y=winners.Tournament)

g.set_xticklabels(labels = winners.Winner,  rotation=90)

plt.title('Grand Slams won since 2000')

plt.show()
winners_slam = wins.groupby(['Winner', 'Tournament']).count()

winners_slam = winners_slam.reset_index()

# winners_slam = winners_slam.sort_values(['Winner'], ascending=False)

winners_slam.columns = ['Winner','Tournament', 'Count']

winners_slam
winners_slam = winners_slam.dropna()
fig = plt.figure(figsize=(15,8))



g = sns.catplot(x="Winner", y="Count", hue = "Tournament", data=winners_slam, kind="bar", size=6, aspect=2)

g.set_xticklabels(labels = winners_slam.Winner.unique(),  rotation=90)

plt.title('Grand Slams won since 2000')

plt.show()
# Nadal loves on surface
# Best players on surface
surface = data[['Surface', 'Winner', 'Loser']]
surface_w = surface[['Surface', 'Winner']]

surface_l = surface[['Surface', 'Loser']]

surface_w.columns = ['Surface', 'Player']

surface_l.columns = ['Surface', 'Player']
surface_w['idx'] = range(1, len(surface_w) + 1)

surface_l['idx'] = range(1, len(surface_l) + 1)
surface_w = surface_w.groupby(['Surface', 'Player']).count()

surface_w = surface_w.reset_index()

surface_w.columns = ['Surface', 'Player', 'Count_Win']



surface_l = surface_l.groupby(['Surface', 'Player']).count()

surface_l = surface_l.reset_index()

surface_l.columns = ['Surface', 'Player', 'Count_Lose']
surface = pd.merge(surface_w, surface_l, on=['Surface', 'Player'])
surface['total_play'] = surface['Count_Win'] + surface['Count_Lose']
surface['perc_win'] = round(surface['Count_Win'] / surface['total_play'],4)*100
surface = surface[surface.total_play > 50]
surface.sort_values(by='perc_win', ascending=False).head(30)
surface.Surface.unique()
# Best player on Clay



top_Clay = surface[surface.Surface == 'Clay'].sort_values(by='perc_win', ascending = False).head(10)

g=sns.catplot(x='Player', y='perc_win', data=top_Clay, kind='bar', size=6, aspect=2)

t=g.set_xticklabels(labels = top_Clay.Player,  rotation=90)

title=plt.title('Best players on Clay')
# Best player on Grass



top_Grass = surface[surface.Surface == 'Grass'].sort_values(by='perc_win', ascending = False).head(10)

g=sns.catplot(x='Player', y='perc_win', data=top_Grass, kind='bar', size=6, aspect=2)

t=g.set_xticklabels(labels = top_Grass.Player,  rotation=90)

title=plt.title('Best players on Grass')
# Best player on Hard



top_Hard = surface[surface.Surface == 'Hard'].sort_values(by='perc_win', ascending = False).head(10)

g=sns.catplot(x='Player', y='perc_win', data=top_Hard, kind='bar', size=6, aspect=2)

g.set_xticklabels(labels = top_Hard.Player,  rotation=90)

title=plt.title('Best players on Hard')
# Best percentage overall



career = data[['Winner', 'Loser']]



career_w = data[['Winner']]

career_l = data[['Loser']]

career_w.columns = ['Player']

career_l.columns = ['Player']



career_w['idx'] = range(1, len(career_w) + 1)

career_l['idx'] = range(1, len(career_l) + 1)



career_w = career_w.groupby('Player').count()

career_w = career_w.reset_index()

career_w.columns = ['Player', 'Count_Win']



career_l = career_l.groupby('Player').count()

career_l = career_l.reset_index()

career_l.columns = ['Player', 'Count_Lose']



career = pd.merge(career_w, career_l, on='Player')



career['total_play'] = career['Count_Win'] + career['Count_Lose']

career['perc_win'] = round(career['Count_Win'] / career['total_play'],4)*100



career = career[career.total_play > 500]



career = career.sort_values(by='perc_win', ascending=False).head(20)

career
g=sns.catplot(x='Player', y='perc_win', data=career, kind='bar', size=7, aspect=2)

g.set_xticklabels(labels = career.Player,  rotation=90)

title=plt.title('Best players overall since 2000')
surface_top3 = surface[(surface.Player.isin(['Federer R.', 'Nadal R.', 'Djokovic N.'])) & (surface.Surface != 'Carpet')]

surface_top3
surface_top3 = pd.pivot_table(surface_top3, values='perc_win', columns=['Surface'], index=['Player'])

surface_top3.index.names

# surface_top3.columns = ['Player', 'Clay', 'Grass', 'Hard']
surface_top3[surface_top3.index == "Federer R."]
# Radar chart for Surface

%matplotlib inline



labels = np.array(['Clay', 'Grass', 'Hard'])

federer = surface_top3.loc[surface_top3[surface_top3.index == "Federer R."].index[0],labels].values
federer = surface_top3.loc[surface_top3[surface_top3.index == "Federer R."].index[0],labels].values

nadal = surface_top3.loc[surface_top3[surface_top3.index == "Nadal R."].index[0],labels].values

djokovic = surface_top3.loc[surface_top3[surface_top3.index == "Djokovic N."].index[0],labels].values
wins_top3 = pd.DataFrame([federer, nadal, djokovic])

wins_top3.columns = ['Clay', 'Grass', 'Hard']

wins_top3['Player'] = ['Federer R.', 'Nadal R.', 'Djokovic N.']

wins_top3 = wins_top3[['Player', 'Clay', 'Grass', 'Hard']]
federer = np.concatenate((federer,[federer[0]]))

nadal = np.concatenate((nadal,[nadal[0]]))

djokovic = np.concatenate((djokovic,[djokovic[0]]))
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)



# close the plot

angles=np.concatenate((angles,[angles[0]]))
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(111, polar=True)

ax1.plot(angles, federer, 'o-', linewidth=2, label = 'Federer')

ax1.fill(angles, federer, alpha=0.25)

ax1.set_thetagrids(angles * 180/np.pi, labels)

ax1.grid(True)



ax2 = fig.add_subplot(111, polar=True)

ax2.plot(angles, nadal, 'o-', linewidth=2, label = 'Nadal')

ax2.fill(angles, nadal, alpha=0.25)

ax2.set_thetagrids(angles * 180/np.pi, labels)

ax2.grid(True)



ax3 = fig.add_subplot(111, polar=True)

ax3.plot(angles, djokovic, 'o-', linewidth=2, label = 'Djokovic')

ax3.fill(angles, djokovic, alpha=0.25)

ax3.set_thetagrids(angles * 180/np.pi, labels)

ax3.grid(True)



l=plt.legend(bbox_to_anchor=(1.1,1))
wins_top3['mean_surface'] = wins_top3.iloc[:, 1:].sum(axis=1) /3
wins_top3.sort_values(by='mean_surface', ascending=False)
# The most complete player is Djokovic