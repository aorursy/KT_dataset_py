import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import datetime as dt

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/data.csv")
data.head()
data.isnull().any()
some_clubs = ('Juventus', 'Real Madrid', 'Paris Saint-Germain', 'Ajax', 'LA Galaxy', 'Manchester United')

df_club = data.loc[data['Club'].isin(some_clubs) & data['Age']]



fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

ax = sns.violinplot(x="Club", y="Age", data=df_club);

sns.swarmplot(x='Club', y='Age',color='k', data=df_club);

ax.set_title(label='Distribution of age in some clubs', fontsize=20);
some_count = ('Brazil', 'France', 'England', 'India', 'Egypt', 'Japan','Cameroon')

df_club = data.loc[data['Nationality'].isin(some_count) & data['Age']]



fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

ax = sns.boxplot(x="Nationality", y="Age", data=df_club);

ax.set_title(label='Distribution of age in some countries', fontsize=20);
sns.set(style ="dark", palette="colorblind", color_codes=True)

x = data.Age

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins = 58, kde = False, color='g')

ax.set_xlabel(xlabel="Player\'s age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize=16)

ax.set_title(label='Histogram of players age', fontsize=20)

plt.show()
pos_coun=data.groupby('Position')['ID'].count()

pos_coun=pos_coun.sort_values(ascending = False)

pos_coun=pd.DataFrame(pos_coun)
pos_coun.head(10)
fig, ax = plt.subplots(figsize=(12,5))

sns.barplot(x = pos_coun.index, y = 'ID', data = pos_coun)

plt.xticks(rotation=20)

ax.set_xlabel(xlabel="Player\'s age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize=16)

ax.set_title(label='Bar graph of Player positions', fontsize=20)
data.groupby(data['Position'])['Overall'].idxmax()
from IPython.display import display, HTML

display(HTML(data.iloc[data.groupby(data['Position'])['Overall'].idxmax()][['Name', 'Position']].to_html(index=False)))
data.groupby(data['Position'])['Overall'].idxmax()
plt.figure(figsize=(14,7))

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

N=18207

colors = np.random.rand(N)

ax=plt.scatter(data.Vision, data['LongPassing'],c=colors, alpha=0.5)

#ax.set_title(label='Relation Vision and Long Passing with respected Skills of players', fontsize=20);
player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

)



# Top three features per position

for i, val in data.groupby(data['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i, *tuple(val.nlargest(3).index)))
sns.jointplot(x=data['Vision'], y=data['LongPassing'], kind="hex", color="#2CB380");
sns.jointplot(x=data['Agility'], y=data['Reactions'], kind="kde", color="y");
club_ov=data.groupby('Club')['Overall'].mean()
club_ov=club_ov.sort_values(ascending=False)

club_ov
some_clubs = ('LA Galaxy', 'Manchester United')

wage_pl = data.loc[data['Club'].isin(some_clubs) & data['Wage']]

wage_pl=wage_pl
wage=wage_pl.groupby(['Club','Wage','Name'])['Name'].count()

wage
weak=data.groupby('Weak Foot')['ID'].count()

weak=pd.DataFrame(weak)

weak
fig, ax = plt.subplots(figsize=(8,7))

explode = (0.01,0.04,0.05,0.07,0.09)

ax.pie(weak.ID, labels = None, autopct='%1.1f%%', startangle=90, shadow = True, explode = explode)

ax.legend(bbox_to_anchor=(1,0.5), labels=weak.index)
loan_from=data.groupby('Loaned From')['ID'].count()

loan_from=loan_from.sort_values(ascending=False)

loan_from=pd.DataFrame(loan_from)

loan_from=loan_from.head(10)
loan_from.index.names=['Team']

loan_from
fig, ax = plt.subplots(figsize=(12,5))



plt.xticks(rotation=20)

sns.barplot(x=loan_from.index, y="ID", data=loan_from)
sns.jointplot(x=data['Stamina'], y=data['SprintSpeed'], kind="hex", color="r");
intce=data[['Name','Interceptions']].sort_values('Interceptions',ascending=False)

intce.head()
some_clubs = ('Juventus', 'Real Madrid', 'Paris Saint-Germain', 'Ajax', 'LA Galaxy', 'Manchester United')

df_club = data.loc[data['Club'].isin(some_clubs) & data['Interceptions']]



fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

ax = sns.boxenplot(x="Club", y="Interceptions", data=df_club);

ax.set_title(label='Distribution of Interceptions in some clubs', fontsize=20);