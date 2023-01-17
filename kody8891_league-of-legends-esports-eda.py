import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

kills=pd.read_csv('../input/leagueoflegends/kills.csv')

structures=pd.read_csv('../input/leagueoflegends/structures.csv')

monsters=pd.read_csv('../input/leagueoflegends/monsters.csv')

matches=pd.read_csv('../input/leagueoflegends/matchinfo.csv')

gold=pd.read_csv('../input/leagueoflegends/gold.csv')

bans=pd.read_csv('../input/leagueoflegends/bans.csv')

descrpition=pd.read_csv('../input/leagueoflegends/_columns.csv')

wins=pd.read_csv('../input/leagueoflegends/LeagueofLegends.csv')
kills.dtypes
kills.isna().sum()

# totally expect many of the assists values to be NA because of the nature of the stat. so only drop the 113 rows where missing  

kills=kills.dropna(subset = ['Time'])

kills.isna().sum()

kills[kills[['x_pos']].apply(lambda x: x[0].isdigit(), axis=1)]

# kills by game time 

kills['Time'].astype(int)

sns.distplot(kills.Time,bins=30)
kills = kills[kills.x_pos != 'TooEarly']

kills = kills[kills.y_pos != 'TooEarly']
assists=pd.melt(kills,id_vars=['Assist_1'],value_vars=['Assist_2', 'Assist_3','Assist_4'])
plt.figure(figsize=(9,5))

plt.title('Most Kills 2015-2017', fontsize=20)

plt.xticks(rotation=30)

sns.countplot(kills.Killer,order=kills.Killer.value_counts().iloc[:10].index)

plt.show()



plt.figure(figsize=(9,5))

plt.title('Most Deaths 2015-2017', fontsize=20)

plt.xticks(rotation=30)

sns.countplot(kills.Victim,order=kills.Victim.value_counts().iloc[:10].index)

plt.show()



plt.figure(figsize=(9,5))

plt.title('Most Assists 2015-2017', fontsize=20)

plt.xticks(rotation=30)

sns.countplot(assists.Assist_1,order=assists.Assist_1.value_counts().iloc[:10].index)

plt.show()





kda=(kills.Killer.value_counts() + assists.Assist_1.value_counts()) / kills.Victim.value_counts()

kda.sort_values(ascending=False).iloc[:10].plot(kind='bar')
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

sns.kdeplot(kills.x_pos, kills.y_pos, cmap=cmap, n_levels=60, shade=True)
monsters['Team'].astype(str)

monsters.Team.unique()

#dragon split

dragon=monsters[(monsters.Team == 'bDragons') | (monsters.Team == 'rDragons')]



# baron split

hearld=monsters[(monsters.Team == 'bHeralds') | (monsters.Team == 'rHeralds')]



# hearld split

baron=monsters[(monsters.Team == 'bBarons') | (monsters.Team == 'rBarons')]

dragon.Team.value_counts()

baron.Team.value_counts()
hearld.Team.value_counts()
values=[14127,14658]

my_labels = ['Blue Side','Red Side ']

colors=['b','r']

plt.pie(values,labels=my_labels,colors=colors,autopct='%1.1f%%')

plt.title('Dragons by Side')

plt.axis('equal')

plt.show()



values2=[5524,5571]

plt.pie(values2,labels=my_labels,colors=colors,autopct='%1.1f%%')

plt.title('Barons by Side')

plt.axis('equal')

plt.show()



values3=[2481,1887]

plt.pie(values3,labels=my_labels,colors=colors,autopct='%1.1f%%')

plt.title('Rift Heralds by Side')

plt.axis('equal')

plt.show()
matches.groupby(['bResult','rResult']).size().reset_index().groupby('rResult')[[0]].max()
values3=[4146,3474]

my_labels = ['Blue Side','Red Side ']

colors=['b','r']

plt.pie(values3,labels=my_labels,colors=colors,autopct='%1.1f%%')

plt.title('Wins by Side')

plt.axis('equal')

plt.show()

towers=structures[(structures.Team == 'bTowers') | (structures.Team == 'rTowers')]

sns.countplot(towers.Team,order=towers.Team.value_counts().iloc[:10].index)

plt.show()
gold.isna().sum()

gold=gold.dropna()

gold.head()
ax=matches.groupby(['blueTopChamp', 'redTopChamp']).size().sort_values(ascending=False).iloc[:10].plot(kind='bar')

ax.set_title('Top Lane Matchups: League of Tanks', fontsize=20)  
ax=matches.groupby(['blueJungleChamp', 'redJungleChamp']).size().sort_values(ascending=False).iloc[:10].plot(kind='bar')

ax.set_title('Jungle Matchups: Variety', fontsize=20)
ax=matches.groupby(['blueMiddleChamp', 'redMiddleChamp']).size().sort_values(ascending=False).iloc[:10].plot(kind='bar')

ax.set_title('Midlane Matchups: Mages', fontsize=20)
ax=matches.groupby(['blueADCChamp', 'redADCChamp']).size().sort_values(ascending=False).iloc[:10].plot(kind='bar')

ax.set_title('ADC Matchups: Utility', fontsize=20)
ax=matches.groupby(['blueSupportChamp', 'redSupportChamp']).size().sort_values(ascending=False).iloc[:10].plot(kind='bar')

ax.set_title('Support Matchups: Definetely Tank Meta', fontsize=20)  
plt.figure(figsize=(8,4)) # this creates a figure 8 inch wide, 4 inch high

x=sns.countplot(x='redTopChamp',data=matches,order=matches.redTopChamp.value_counts().iloc[:10].index)

x.set_title('Most Banned Top Champs 2015-2017')

plt.show()



plt.figure(figsize=(8,4)) # this creates a figure 8 inch wide, 4 inch high

x=sns.countplot(x='redJungleChamp',data=matches,order=matches.redJungleChamp.value_counts().iloc[:10].index)

x.set_title('Most Banned Jungle Champs 2015-2017')

plt.show()



plt.figure(figsize=(8,4)) # this creates a figure 8 inch wide, 4 inch high

x=sns.countplot(x='redMiddleChamp',data=matches,order=matches.redMiddleChamp.value_counts().iloc[:10].index)

x.set_title('Most Banned Mid Champs 2015-2017')

plt.show()



plt.figure(figsize=(8,4)) # this creates a figure 8 inch wide, 4 inch high

x=sns.countplot(x='redADCChamp',data=matches,order=matches.redADCChamp.value_counts().iloc[:10].index)

x.set_title('Most Banned ADC Champs 2015-2017')

plt.show()



plt.figure(figsize=(8,4)) # this creates a figure 8 inch wide, 4 inch high

x=sns.countplot(x='redSupportChamp',data=matches,order=matches.redSupportChamp.value_counts().iloc[:10].index)

x.set_title('Most Banned Support Champs 2015-2017')

plt.show()
# of matches by league

wins.League.value_counts().plot(kind='bar')

plt.title('# of Matches by League')
major_region=['LCK','NALCS','EULCS']

major= wins[wins.League.isin(major_region)]



tournaments=['WC','IEM','MSI']

tourney= wins[wins.League.isin(tournaments)]

plt.figure(figsize=(9,10))

sns.catplot(x="League", y="gamelength",kind='boxen', data=major)

plt.title('Major Region Game Length')
plt.figure(figsize=(9,10))

sns.catplot(x="League", y="gamelength",kind='boxen', data=tourney)

plt.title('International Tournaments Game Length')
plays = wins[wins['blueTopChamp'].map(wins['blueTopChamp'].value_counts()) >49]

plays.head()
# blue top champ by average of bResult, order by champion wr%

topwr=plays.groupby(['blueTopChamp'])['bResult'].mean().sort_values(ascending=False).to_frame()

jungwr=plays.groupby(['blueJungleChamp'])['bResult'].mean().sort_values(ascending=False).to_frame()

midwr=plays.groupby(['blueMiddleChamp'])['bResult'].mean().sort_values(ascending=False).to_frame()

adcwr=plays.groupby(['blueADCChamp'])['bResult'].mean().sort_values(ascending=False).to_frame()

suppwr=plays.groupby(['blueSupportChamp'])['bResult'].mean().sort_values(ascending=False).to_frame()

topwr
jungwr
midwr
adcwr
suppwr
unique_comps=plays.groupby(['blueTopChamp','blueJungleChamp','blueMiddleChamp','blueADCChamp','blueSupportChamp'])['bResult'].mean().sort_values(ascending=False).to_frame()

unique_comps
# of games with unique blue side comps 

len(unique_comps)/len(wins)