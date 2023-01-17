# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()
from urllib.request import urlretrieve



#Assign url of files:url1, url2

url1='https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Projects/matches.csv'



#save file locally

urlretrieve(url1, 'matches.csv')



df1=pd.read_csv('matches.csv')

df1.head()
from urllib.request import urlretrieve



#Assign url of files:url1, url2

url2='https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Projects/deliveries.csv'



#save file locally

urlretrieve(url2, 'deliveries.csv')



df2=pd.read_csv('deliveries.csv')

df2.head()
df1=df1.replace({'Mumbai Indians':'MI','Royal Challengers Bangalore':'RCB', 'Kolkata Knight Riders':'KKR', 'Kings XI Punjab':'KXIP'

     , 'Delhi Daredevils':'DD', 'Chennai Super Kings':'CSK', 'Rajasthan Royals':'RR', 'Sunrisers Hyderabad':'SRH'

     , 'Deccan Chargers':'DD', 'Pune Warriors':'PWR', 'Gujarat Lions':'GL', 'Rising Pune Supergiant':'RPSG'

     , 'Rising Pune Supergiants':'RPSG', 'Kochi Tuskers Kerala':'KTK'})



df1.head()
print(df1.shape)
df1.info()
plt.figure(figsize=(20,6))

sns.countplot(x='season', data=df1, palette='winter')

plt.xlabel('Season', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.title('Total matches played in each Season', fontsize=16)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(y='city', data=df1, palette='winter', order=df1.city.value_counts().iloc[:15].index)



plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('Count', fontsize=11)

plt.ylabel('City', fontsize=11)

plt.title('City hosted Most IPL matches', fontsize=16)

plt.axis('tight')



plt.show()
plt.figure(figsize=(12,6))



fav_ground = df1['venue'].value_counts().reset_index()

fav_ground.columns = ['venue','count']

sns.barplot(x = 'count',y = 'venue', data = fav_ground[:10], palette = 'Blues_d')

plt.xticks(fontsize=11)

plt.yticks(fontsize=11)

plt.xlabel('Count', fontsize=11)

plt.ylabel('Stadium', fontsize=11)



plt.title('Stadium hosted Most IPL matches', fontsize=16)

plt.axis('tight')



plt.show()
player_of_match= df1['player_of_match'].value_counts().reset_index()

player_of_match.columns= ['Player Name', 'Matches']

player_of_match=player_of_match[:11].sort_values('Matches')

player_of_match.head()
#MVP player can be determined from the number of  times they got MOM



plt.figure(figsize=(20,6))

plt.barh(player_of_match['Player Name'], player_of_match['Matches'], align='center')

plt.yticks(fontsize=14)

plt.ylabel('Player', fontsize=13)

plt.xlabel('No of Matches', fontsize= 13)

plt.xticks([0,4,8,12,16,20])

plt.title('Most Man of the Match awards through all seasons', fontsize=16)

plt.show()
#Lets findout how Toss played major role in Team's wins or losses

toss=df1.groupby(['season', 'toss_winner', 'toss_decision'])['winner'].value_counts().reset_index(name='count')

toss['result']= np.where(toss['toss_winner']== toss['winner'], 'won', 'lost')

toss.head()
plt.figure(figsize=(20,6))

sns.countplot(x='winner', hue='toss_decision', data=toss, palette='dark')

plt.xlabel('Team', fontsize=16)

plt.ylabel('No of Matches', fontsize= 16)

plt.title('Teams decision', fontsize=16)

plt.show()
toss_decider= df1.groupby('season')['toss_decision'].value_counts().unstack().reset_index()

toss_decider
plt.figure(figsize=(20,6))

plt.plot(toss_decider['season'],toss_decider['bat'], color='orange', marker='d', label= 'batting first')

plt.plot(toss_decider['season'], toss_decider['field'], color='blue', marker='o', label=' batting second')

plt.xlabel('Year', fontsize= 14)

plt.ylabel('Matches', fontsize=14)

plt.xticks([2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018])

plt.legend(fontsize=12)

plt.title('Toss Decision trend over the season', fontsize=18)

plt.show()
#Lets explore which team have Maximum win

matches_played=pd.concat([df1['team1'],df1['team2']])

matches_played=matches_played.value_counts().reset_index()

matches_played.columns=['Team','Total Matches']

matches_played['wins']=df1['winner'].value_counts().reset_index()['winner']





print(matches_played)
matches_played['win%'] = round(matches_played['wins']/matches_played['Total Matches'] * 100, 1)

matches_played= matches_played.sort_values('win%', ascending=False)

matches_played
plt.figure(figsize=(16,6))

plt.bar(matches_played['Team'], matches_played['win%'], align='center')

plt.xlabel('Team', fontsize=14)

plt.ylabel('Winning Percentage', fontsize= 14)

plt.title('Maximum win percentage', fontsize=16)

plt.show()
print(df2.shape)
df2.info()
df2=df2.replace({'Mumbai Indians':'MI','Royal Challengers Bangalore':'RCB', 'Kolkata Knight Riders':'KKR', 'Kings XI Punjab':'KXIP'

     , 'Delhi Daredevils':'DD', 'Chennai Super Kings':'CSK', 'Rajasthan Royals':'RR', 'Sunrisers Hyderabad':'SRH'

     , 'Deccan Chargers':'DD', 'Pune Warriors':'PWR', 'Gujarat Lions':'GL', 'Rising Pune Supergiant':'RPSG'

     , 'Rising Pune Supergiants':'RPSG', 'Kochi Tuskers Kerala':'KTK'})



df2.head()
df2.columns
team_score = df2.groupby(['match_id', 'inning'])['total_runs'].sum().unstack().reset_index()

team_score.columns= ['match_id', '1st Innning score', '2nd Innning score', 'Team1 Superover score', 'Team2 Superover score']



matches_agg= pd.merge(df1, team_score, left_on='id', right_on='match_id', how='outer')

matches_agg
runs_df = df2.groupby('batsman')['batsman_runs'].agg('sum').reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

runs_df= runs_df[:10].sort_values('batsman_runs', ascending=False)



plt.figure(figsize=(14,6))

sns.barplot(x='batsman', y='batsman_runs', data=runs_df, palette='spring')

plt.xlabel('Batsman', fontsize=14)

plt.ylabel('Total Runs', fontsize= 14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title('IPL Top Scorer', fontsize=16)

plt.show()
#merge two dataframes to analysis on Batsmen

batsmen = df1[['id','season']].merge(df2, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)

Total_6s = batsmen.groupby(['season'])['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index()

Total_4s = batsmen.groupby(['season'])['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index()
Runs_scored= Total_6s.merge(Total_4s, on='season')

Runs_scored.columns= ['Season', 'Sixes', 'Fours']

Runs_scored
max_6s = df2.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

max_6s = max_6s.iloc[:10,:]

max_6s.columns= ['Batsmen', 'Maximum 6s']



plt.figure(figsize=(14,6))

sns.barplot(x='Batsmen', y='Maximum 6s', data=max_6s, palette='spring')

plt.xlabel('Batsman', fontsize=14)

plt.ylabel('No of Sixes', fontsize= 14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title('Batsmen with most Sixes', fontsize=16) 

plt.show()
max_4s = df2.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index().sort_values(by='batsman_runs', ascending=False).reset_index(drop=True)

max_4s = max_4s.iloc[:10,:]

max_4s.columns= ['Batsmen', 'Maximum 4s']



plt.figure(figsize=(14,6))

sns.barplot(x='Batsmen', y='Maximum 4s', data=max_4s, palette='spring')

plt.xlabel('Batsman', fontsize=14)

plt.ylabel('No of Fours', fontsize= 14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title('Batsmen with most Fours', fontsize=16)

plt.show()
plt.figure(figsize=(12,6))

sns.countplot(y='dismissal_kind', data=df2)

plt.ylabel('Dismissals', fontsize= 14)

plt.show()
bowler_df = df2.groupby('bowler')['ball'].agg('count').reset_index().sort_values(by='ball', ascending=False).reset_index(drop=True)

bowler_df = bowler_df.iloc[:10,:]



bowler_df
plt.figure(figsize=(14,6))

sns.barplot(x='bowler', y='ball', data=bowler_df, palette='winter')

plt.xlabel('Bowler', fontsize=14)

plt.ylabel('Balls Bowled', fontsize= 14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title('Bowlers with maximum balls bowled', fontsize=16)

plt.show()