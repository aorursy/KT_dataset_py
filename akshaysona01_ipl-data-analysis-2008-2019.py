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
# Importing all the basic libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
match_data = pd.read_csv('../input/ipldata/matches.csv') # Importing datafile by using location. As we know it is csv file 
match_data # visualize data in the following table
match_data.info()  # We get basic information about data like null values count or datatypre column name etc.

# As umpire column is not important for our datset, so removing columns
match_data.drop(columns=['umpire1','umpire2','umpire3'], inplace=True)
match_data.shape #shape is used for getting column and row size
match_data.columns #Getting all column names in the dataset

match_data.team1.unique()
match_data.team1.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant', 'Delhi Daredevils':'Delhi Capitals'},inplace=True) 
# Same team but with different name. So, we replaced it withits real and unique name
match_data.team2.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant', 'Delhi Daredevils':'Delhi Capitals'},inplace=True)
# Same team but with different name. So, we replaced it withits real and unique name
match_data.winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant', 'Delhi Daredevils':'Delhi Capitals'},inplace=True)
# Same team but with different name. So, we replaced it withits real and unique name
# We can see that, there are many rows in the dataset where city is not mentioned or simply NaN. We need to find it out 
match_data[match_data.city.isna()]
match_data.loc[[461,462,466,468,469,474,476],'city'] = 'Dubai' #As all the null cities venue is Dubai, we are going fill it with Dubai
match_data.loc[[461,462,466,468,469,474,476]]  # Here we can check NaN is filled with Dubai.
match_data.info() 

match_data[match_data.winner.isna()] # Checking where there is NaN in winner column
match_data.city.replace({'Bangalore' : 'Bengaluru'},inplace=True) # Replacing Bangalore to Begaluruby using replace function
match_data.city.unique() # Here, we can check whether there is any mistake in city name
match_data
match_data.id.count()
num_of_matches = match_data.groupby('season')[['id']].count() # Grouping all the matches and season 

plt.title('Matches played in each season') # Plot table with given title
plt.xlabel('Year')  # plotting x axis
plt.ylabel('Count of matches') #Plotting y axis
plt.xticks(num_of_matches.index) # importing number of matches played each year
plt.bar(num_of_matches.index,num_of_matches.id,width=0.4, alpha=0.8);
winner_team = match_data.groupby('winner')[['id']].count()
winner_team = winner_team.sort_values('id', ascending=False).reset_index()
winner_team.rename(columns = { 'id': 'matches_won','winner':'team'}, inplace = True)
winner_team
plt.title('Number of matches won by each team',fontweight=800)
plt.xlabel('Teams')
plt.ylabel('Total no. of matches won')
plt.xticks(rotation=90,fontsize=10)
plt.bar(winner_team.team,winner_team.matches_won, alpha=0.8)
# Inference 4: Number of matches hosted by each city
city_name = match_data.groupby('city')[['id']].count()
city_name=city_name.sort_values('id', ascending=False).reset_index()
city_name.rename(columns = { 'id': 'matches'}, inplace = True)

plt.figure(figsize=(20, 15))
plt.title('Number of matches hosted by each city')
sns.barplot(y='city', x='matches', data=city_name);
final_match_win = match_data.groupby('season').tail(1).sort_values('season').reset_index()
final_match_win['winner'].value_counts()
plt.title('Count of Season Winners')
sns.countplot(x=final_match_win['winner']);
second_runnerup = pd.concat([final_match_win['team1'],final_match_win['team2']])
second_runnerup.value_counts()
plt.title('Second Runner up Team')
sns.countplot(x=second_runnerup.value_counts());
mom_count =match_data.groupby('player_of_match')[['id']].count()
mom_count = mom_count.sort_values('id',ascending=False).head(10)
mom_count
matches_team = pd.concat([match_data['team1'],match_data['team2']])
matches_team_df=matches_team.value_counts().reset_index()  
matches_team_df.columns=['team','total_matches']   
matches_team_df.set_index('team',inplace=True)     
merged_stats_df=matches_team_df.merge(winner_team,on='team')
merged_stats_df['winning_percent'] = (merged_stats_df.matches_won/merged_stats_df.total_matches)*100
merged_stats_df
mi_csk=match_data[((match_data.team1 =="Mumbai Indians") & (match_data.team2 =="Chennai Super Kings")) | 
                  ((match_data.team2 =="Mumbai Indians") & (match_data.team1 =="Chennai Super Kings"))]
mi_csk['winner'].value_counts()
team_count=[];
for i in range(2008,2020):
    team_count.append(match_data[match_data.season == i].team1.unique().shape[0])

sns.barplot(x=np.arange(2008,2020),y=team_count)
plt.title('Number of teams in each season')
plt.xlabel('Seasons')
plt.ylabel('No. of teams')
plt.yticks(np.arange(0,11));
match_toss_winner_df = match_data[match_data['toss_winner']==match_data['winner']]

match_toss_winner_df = match_toss_winner_df[match_toss_winner_df.result != 'tie']

count_toss_winner_decision_df = match_toss_winner_df['toss_decision'].value_counts() 

toss_loser_match_winner_df = match_data[match_data['toss_winner']!=match_data['winner']] 

toss_loser_match_winner_df = toss_loser_match_winner_df[toss_loser_match_winner_df.result != 'tie']

count_toss_decision_loser_df = toss_loser_match_winner_df['toss_decision'].value_counts()

team_bat_first_won =count_toss_decision_loser_df.field+count_toss_winner_decision_df.bat

print('Total number of matches where team batting first is the winner of the match is {}'.format(team_bat_first_won))
