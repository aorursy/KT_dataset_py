#Loading libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls


#Importing the datasets 

players = pd.read_csv("../input/Player.csv", encoding='ISO-8859-1' )
player_match = pd.read_csv("../input/Player_match.csv", encoding='ISO-8859-1' )
team = pd.read_csv("../input/Team.csv", encoding='ISO-8859-1' )
ball_fact = pd.read_csv ("../input/Ball_By_Ball.csv", encoding='ISO-8859-1' )
match = pd.read_csv ("../input/Match.csv", encoding='ISO-8859-1' )



players.head(2)
player_match.describe(include='all')
captain = player_match[player_match['Role_Desc'] == 'Captain']
captain.Player_Name.unique()
plt.figure(figsize=(14,6))
sns.countplot(x='Age_As_on_match',data=player_match)
Keepers = player_match[player_match['Role_Desc'] == 'Captain']
Keepers.Player_Name.unique()
#player_match.describe(include='all')
#Most man of the matches awards
ManofMatch = match.groupby(['ManOfMach']).count()['match_winner']
ManOfMatch_count = ManofMatch.sort_values(axis=0, ascending=False)
ManOfMatch_count.head()
#number of matches per season
plt.figure(figsize=(8,7))
sns.countplot(x='Season_Year', data=match) 

#Number of matches per venue 
plt.figure(figsize=(14,8))
sns.countplot(x='Venue_Name', data=match, order=pd.value_counts(match['Venue_Name']).index) 
plt.xticks(rotation='vertical')
plt.show()
#Wins per team
plt.figure(figsize=(8,6))
ax=sns.countplot(x='match_winner', data=match, order=pd.value_counts(match['match_winner']).index) 
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
#Toss wins per team
plt.figure(figsize=(8,6))
ax=sns.countplot(x='Toss_Winner', data=match, order=pd.value_counts(match['Toss_Winner']).index) 
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
match.replace(to_replace='Field', value = 'field', inplace=True) #Replace 'Field' with 'field'
match.replace(to_replace='Bat', value = 'bat', inplace=True) #Replace 'Bat' with 'bat'
plt.figure(figsize=(12,6))
sns.countplot(x='Season_Year', hue='Toss_Name', data=match)
plt.figure(figsize=(8,6))
sns.countplot(x='Batting_hand', data=players)
plt.figure(figsize=(12,6))
ax=sns.countplot(x='Bowling_skill', data=players, order=pd.value_counts(players['Bowling_skill']).iloc[:10].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()
agg = players['Country_Name'].value_counts()[:10]
labels = list(reversed(list(agg.index )))
values = list(reversed(list(agg.values)))

trace1 = go.Pie(labels=labels, values=values, marker=dict(colors=['red']))
layout = dict(title='Top Countries', legend=dict(orientation="h"));


fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')
ball_fact.describe(include='all')
ball_fact.replace(to_replace='Wides', value = 'wides', inplace=True) #Replace Wides with wides
ball_fact.replace(to_replace='Legbyes', value = 'legbyes', inplace=True) #Legbyes with legbyes
ball_fact.replace(to_replace='Noballs', value = 'noballs', inplace=True) #Noballs with noballs
ball_fact.replace(to_replace='Byes', value = 'byes', inplace=True) #Byes with byes
ball_df = ball_fact[ball_fact['Extra_Type']  != 'No Extras']
agg = ball_df['Extra_Type'].value_counts()[:10]
labels = list(reversed(list(agg.index )))
values = list(reversed(list(agg.values)))

trace1 = go.Bar(x=values, y=labels, opacity=0.75, orientation='h', name="month", marker=dict(color='rgba(0, 20, 50, 0.6)'))
trace1 = go.Pie(labels=labels, values=values)
layout = dict(title='Extras given', legend=dict(orientation="h"));


fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')
ball_df = ball_fact[ball_fact['Out_type']  != 'Not Applicable']
agg = ball_df['Out_type'].value_counts()[:10]
labels = list(reversed(list(agg.index )))
values = list(reversed(list(agg.values)))

trace1 = go.Bar(x=values, y=labels, opacity=0.75, orientation='h', name="month", marker=dict(color='rgba(0, 20, 50, 0.6)'))
trace1 = go.Pie(labels=labels, values=values)
layout = dict(title='Out Type', legend=dict(orientation="h"));


fig = go.Figure(data=[trace1], layout=layout)
iplot(fig, filename='stacked-bar')
