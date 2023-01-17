# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline     

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/ipldata/matches.csv')
matches = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
print(f'Number of rows    = {len(matches)}')
print(f'Number of columns = {len(matches.columns)}')
matches.head()
print(f'Number of rows    = {len(df)}')
print(f'Number of columns = {len(df.columns)}')
df.head()
print(df.describe())
print(matches.describe())#Statistical information of datset
df.info()
matches.info()  #Getting the information about the matches dataframe
matches.isnull().sum()
df.isnull().sum()
matches.columns
df.columns
#Dropping the column of umpires as we will not be using it for any data analysis
df.drop(columns=['umpire1','umpire2'], inplace=True)
#Finding the shape of the matches dataframe
df.shape
#Finding the columns of matches dataframe
df.columns
df.team1.unique()
df.isnull()
#Using replace method in pandas library

df.team1.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant', 'Delhi Daredevils':'Delhi Capitals'},inplace=True)
df.team2.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant', 'Delhi Daredevils':'Delhi Capitals'},inplace=True)
df.winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant', 'Delhi Daredevils':'Delhi Capitals'},inplace=True)
df.team1.unique()
#Finding the rows where the city is NaN
df[df.city.isna()]
df.sample(6)
#placing city value to Dubai where there is no value
df.loc[[461,462,466,468,469,474,476],'city'] = 'Dubai'
df.loc[[461,462,466,468,469,474,476]]
df[df.result != 'normal']
#Finding the rows where the winner is NaN
df[df.winner.isna()]
#Making Bangalore as Bengaluru
df.city.replace({'Bangalore' : 'Bengaluru'},inplace=True)
df.city.unique()
#Finding the number of IPL matches played till 2019
df.id.count()
#Finding the number of teams who played IPL till 2019
print(df.team1.unique())
print('{} different teams played the IPL from 2008-2019'.format(df.team1.unique().shape[0]))
#NUMBER OF MATCHES IN EACH SEASON
num_matches_df = df.groupby('season')[['id']].count()
num_matches_df
#Number of matches in each season
plt.title('Number of matches in each season',fontweight=800)
plt.xlabel('Seasons')
plt.ylabel('Total no. of matches')
plt.xticks(num_matches_df.index)
plt.bar(num_matches_df.index,num_matches_df.id,width=0.8, color=['#15244C','#FFFF48','#292734','#EF2920','#CD202D','#ECC5F2',
               '#294A73','#242307','#158EA6','#E82865',
               '#005DB7','#C23E25'], alpha=0.8);
#Number of matches won by each team from 2008-2019

winner_df = df.groupby('winner')[['id']].count()
winner_df=winner_df.sort_values('id', ascending=False).reset_index()

winner_df.rename(columns = { 'id': 'matches_won','winner':'team'}, inplace = True)
winner_df
#Number of matches in each season
plt.title('Number of matches won by each team from 2008-2019',fontweight=800)
plt.xlabel('Teams')
plt.ylabel('Total no. of matches won')
plt.xticks(rotation=90,fontsize=10)
plt.bar(winner_df.team,winner_df.matches_won, color=['#15244C','#FFFF48','#292734','#EF2920','#CD202D','#ECC5F2',
               '#294A73','#D4480B','#242307','#FD511F','#158EA6','#E82865',
               '#005DB7','#C23E25','#E82878'], alpha=0.8)
#COMPARISON BETWEEN MATCHES PLAYED AND MATCHES WON BY EACH TEAM

#Total Number of matches played by each team
matches_team = pd.concat([df['team1'],df['team2']])
matches_team_df=matches_team.value_counts().reset_index()  
#value_counts() return a Series containing counts of unique values.
#Series.reset_index() function generate a new DataFrame or Series with the index reset

matches_team_df.columns=['team','total_matches']   #Make two Columns Teams and Total Matches
matches_team_df.set_index('team',inplace=True)     #Sets Team as index
merged_stats_df=matches_team_df.merge(winner_df,on='team')
merged_stats_df['winning_percent'] = (merged_stats_df.matches_won/merged_stats_df.total_matches)*100
merged_stats_df
plt.title('Number of matches played vs Number of matches won',fontweight=800)
plt.xlabel('Teams')
plt.ylabel('Total no. of matches/Winning Percent')
plt.xticks(rotation=90)
plt.bar(merged_stats_df.team,merged_stats_df.total_matches,alpha=0.4)
plt.bar(merged_stats_df.team,merged_stats_df.matches_won, alpha=0.4)
plt.plot(merged_stats_df.team,merged_stats_df.winning_percent,'x-r')
plt.legend(['Winning Percent','Matches Played','Matches Won']);
#MATCHES HOSTED BY EACH CITY
city_df = df.groupby('city')[['id']].count()
city_df=city_df.sort_values('id', ascending=False).reset_index()

city_df.rename(columns = { 'id': 'matches'}, inplace = True)
city_df
plt.figure(figsize=(30, 15))
plt.title('Number of matches hosted by each city from 2008-2019',fontweight=800)
sns.barplot(y='city', x='matches', data=city_df);
#Q1. Find the number of matches where toss winner is the match winner ?
match_toss_winner_df = df[df['toss_winner']==df['winner']]

match_toss_winner_df
#Tie Result
match_toss_winner_df = match_toss_winner_df[match_toss_winner_df.result != 'tie']
print('There are {} matches in the IPL played till now where toss winner is the winner of the match'.format(match_toss_winner_df.id.count()))
#Q2. Find the number of matches where team batting first is the match winner ?
count_toss_winner_decision_df = match_toss_winner_df['toss_decision'].value_counts() #toss_winner decisions
toss_loser_match_winner_df = df[df['toss_winner']!=df['winner']] #cases where toss loser is match winner
toss_loser_match_winner_df = toss_loser_match_winner_df[toss_loser_match_winner_df.result != 'tie'] #Removing tie cases
count_toss_decision_loser_df = toss_loser_match_winner_df['toss_decision'].value_counts() #toss_loser decisions-will be reverse of toss_winner
team_bat_first_won =count_toss_decision_loser_df.field+count_toss_winner_decision_df.bat
print('Total number of matches where team batting first is the winner of the match is {}'.format(team_bat_first_won))
#Q3. Total matches played between MI and CSK and which team won most matches ?
mi_csk=df[((df.team1 =="Mumbai Indians") & 
                   (df.team2 =="Chennai Super Kings")) | 
                  
                  ((df.team2 =="Mumbai Indians") & 
                   (df.team1 =="Chennai Super Kings"))]
mi_csk
plt.title('MI vs CSK')
sns.countplot(x=mi_csk['winner'],palette=['#FFFF48','#15244C'])
plt.text(-0.1,9,s=mi_csk['winner'].value_counts()['Chennai Super Kings'], color='white', size=40)
plt.text(0.95,15,s=mi_csk['winner'].value_counts()['Mumbai Indians'], color='white', size=40);
#Q4. Details of season winner and which team won most seasons? Also find the team which played most finals in IPL till 2019.
#Creating a dataframe of final matches
final_match_df = df.groupby('season').tail(1).sort_values('season').reset_index()
final_match_df
final_match_df['winner'].value_counts()
plt.title('Count of Season Winners',fontweight=800)
sns.countplot(x=final_match_df['winner'],palette=['#ECC5F2','#1C2C46','#FFFF48','#F0E1A1','#15244C','#D4480B']);
final_match_team = pd.concat([final_match_df['team1'],final_match_df['team2']])
final_match_team.value_counts()
plt.title('Count of Finalists in IPL',fontweight=800)
plt.xlabel('Teams')
sns.countplot(x=final_match_team,palette=['#FFFF48','#1C2C46','#15244C','#CD202D','#D4480B','#ECC5F2','#EF2920','#F0E1A1','#E776CA'])
plt.ylabel('Final Matches Played')
plt.xticks(rotation=90);
#Q5. Who are the top 10 players with most number of Man of the Match ?
mom_count =df.groupby('player_of_match')[['id']].count()
mom_count = mom_count.sort_values('id',ascending=False).head(10)
mom_count
plt.title("Top 10 Player with Most Man of the Match Awards",fontweight=800 )
sns.barplot(x=mom_count.index,y=mom_count.id, alpha=0.6);
plt.xticks(rotation=90)
plt.yticks(ticks=np.arange(0,25,5))
plt.ylabel('No. of Awards')
plt.xlabel('Players');
#Q6. Why number of matches are greater in 2011,2012 and 2013 ?
team_count=[];
for i in range(2008,2020):
    team_count.append(df[df.season == i].team1.unique().shape[0])

sns.barplot(x=np.arange(2008,2020),y=team_count)
plt.title('Number of teams in each season', fontweight=800)
plt.xlabel('Seasons')
plt.ylabel('No. of teams')
plt.yticks(np.arange(0,11));
each_season_winner = df.groupby('season')['season','winner'].tail(1)
each_season_winner_sort = each_season_winner.sort_values('season',ascending = True)
sns.countplot('winner', data = each_season_winner_sort)
plt.xticks(rotation = 45, ha = 'right')
plt.ylabel('Number of seasons won by any team.')
plt.show()
#Top 10 Cities by Number Of matches
city_counts= df.groupby('city').apply(lambda x:x['city'].count()).reset_index(name='Match Counts')
top_cities_order=city_counts.sort_values(by='Match Counts',ascending=False)
top_cities=top_cities_order[:10]
print('Top 10 Cities with the maximum number of Matches Played:\n',top_cities)
plt.figure(figsize=(7,7))
plt.pie(top_cities['Match Counts'],labels=top_cities['city'],autopct='%1.1f%%', startangle=30)
plt.axis('equal')
plt.title('Top Cities that have hosted IPL Matches',size=10)
#Effect of home ground:
plt.figure(figsize = (12,6))
venue = df[['city','winner','season']]
venue_season = venue[venue['season'] == 2018]
ax = sns.countplot('city', data = venue_season, hue = 'winner' )
plt.xticks(rotation=30, ha = 'right')
plt.ylabel('Number of matches.')
plt.show()
#Season wise match summary of matches won by runs
fig=plt.gcf()
fig.set_size_inches(18.5,10.5)
sns.swarmplot(df['season'],df[df['win_by_runs']!=0]['win_by_runs'],s=10)
plt.xticks(rotation=90,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Season',fontsize=14)
plt.ylabel('Runs',fontsize=14)
plt.title('Season wise match summary of matches won by runs',fontsize=14)
plt.show()
sns.pairplot(df)
sns.pairplot(matches)