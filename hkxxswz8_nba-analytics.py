import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import warnings
import seaborn as sns 
import os
import json 
import folium
warnings.filterwarnings('ignore')
game_data=pd.read_csv('game_data.csv')
player_data=pd.read_csv('player_data.csv')
test_set = pd.read_csv('test_set.csv')
training_set = pd.read_csv('training_set.csv')
player_data[player_data.isnull().any(axis=1)].head()
game_data[game_data.isnull().any(axis=1)].head()
training_set[training_set.isnull().any(axis=1)].head()

print (len(test_set[test_set['Season'] == '2016-17']),'games are from Season 2016-17.')
print (len(test_set[test_set['Season'] == '2017-18']),'games are from Season 2017-18.')
def get_home_team_id (game_id):
    return training_set[training_set['Game_ID']==game_id]['Home_Team'].unique()
def get_away_team_id (game_id):
    return training_set[training_set['Game_ID']==game_id]['Away_Team'].unique()
def get_season(game_id):
    return training_set[training_set['Game_ID']==game_id]['Season'].unique()

new_df = training_set.groupby('Game_ID').agg('sum').reset_index()

new_df['Home_Team']= new_df['Game_ID'].map(lambda x: get_home_team_id(x)[0])
new_df['Away_Team']= new_df['Game_ID'].map(lambda x: get_away_team_id(x)[0])
new_df['Season']= new_df['Game_ID'].map(lambda x: get_season(x)[0])

Team_Ranking_1617 = ['GSW','SAS','HOU','BOS','CLE','LAC','TOR','UTA','WAS','OKC','ATL',
               'MEM','MIL','IND','POR','CHI','MIA','DEN','DET','CHA','NOP',
               'DAL','SAC','NYK','MIN','ORL','PHI','LAL','PHX','BKN'] 

Team_Ranking_1516 = ['GSW','SAS','CLE','TOR','OKC','LAC','MIA','ATL','BOS','CHA',
               'IND','POR','DET','CHI','DAL','MEM','WAS','HOU','UTA','ORL',
               'MIL','SAC','DEN','NYK','NOP','MIN','PHX','BKN','LAL','PHI'] 

new_df.head()
all_team_viewers = new_df[new_df['Season']=='2016-17'].pivot_table(index='Away_Team',columns='Home_Team',values='Rounded Viewers')
all_team_viewers = all_team_viewers.reindex(Team_Ranking_1516)
all_team_viewers = all_team_viewers[Team_Ranking_1516]
fig = plt.figure(1, figsize=(13, 12))
ax = fig.add_subplot(111)
sns.heatmap(all_team_viewers)
all_team_viewers = new_df[new_df['Season']=='2017-18'].pivot_table(index='Away_Team',columns='Home_Team',values='Rounded Viewers')
all_team_viewers = all_team_viewers.reindex(Team_Ranking_1617)
all_team_viewers = all_team_viewers[Team_Ranking_1617]
fig = plt.figure(1, figsize=(13, 12))
ax = fig.add_subplot(111)
sns.heatmap(all_team_viewers)
Price_1617 = {'GSW':215, 'NYK':177,'LAL':139,'CLE':114,'SAS':105,'CHI':95,
             'TOR':95,'DEN':93,'BOS':90,'BKN':89,'HOU':88,'OKC':85,'SAC':78,
             'PHX':75,'DAL':74,'MIN':73,'POR':70,'PHI':70,'WAS':65,'ATL':57,
             'MIA':55,'UTA':55,'MIL':52,'LAC':50,'ORL':49,'CHA':45,'DET':44,
             'MEM':41,'IND':38,'NOP':35}

view_price_1617 =new_df[new_df['Season']=='2016-17'].groupby('Home_Team').agg('sum')
view_price_1617['Price'] = view_price_1617.index.map(lambda x: Price_1617.get(x))

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
plt.scatter(view_price_1617['Price'],view_price_1617['Rounded Viewers'], s=70,c ='red', edgecolors='black', alpha=0.7)
plt.xlabel('Price', fontsize=14)
plt.ylabel('Total Viewership',fontsize=14)
plt.title('Home Team Ticket Price vs. Total Viewership in 2016-17 Season',fontsize=18)

def get_date(game_id):
    return training_set[training_set['Game_ID'] == game_id]['Game_Date'].unique()[0]
lead_change_view = game_data.loc[:,['Game_ID','Lead_Changes']].dropna().groupby('Game_ID').agg('sum')
lead_change_view['Rounded Viewers'] = lead_change_view.index.map(lambda x: new_df[new_df['Game_ID']==x] \
                                        ['Rounded Viewers'].unique()[0])


fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
plt.hist(lead_change_view['Lead_Changes'],bins=range(0,35,1),edgecolor='black')
plt.xlabel('Lead Changes', fontsize=14)
plt.ylabel('Frequency',fontsize=14)
plt.title('Histogram of Numbers of Lead Changes',fontsize=18)
fig = plt.figure(figsize=(16,11))
ax = fig.add_subplot(111)
plt.scatter(lead_change_view['Lead_Changes'],lead_change_view['Rounded Viewers'], s=50,c ='blue', edgecolors='black', alpha=0.7)
plt.xlabel('Lead Changes', fontsize=14)
plt.ylabel('Total Viewership',fontsize=14)
plt.title('Scatterplot of Lead Changes vs. Total  Viewership',fontsize=18)

score_diff = game_data.dropna().loc[:,['Game_ID','Location','Final_Score']]
# convert the score of one team to minus 
for i in score_diff.index:
    if score_diff.loc[i,'Location'] == 'A':
        score_diff.loc[i,'Final_Score'] = score_diff.loc[i,'Final_Score']*-1

# calculate the score difference

score_diff = score_diff.groupby('Game_ID').agg('sum')
score_diff['Final_Score'] = score_diff['Final_Score'].map(lambda x: abs(x))
score_diff['Rounded Viewers'] = score_diff.index.map(lambda x: new_df[new_df['Game_ID']==x] \
                                        ['Rounded Viewers'].unique()[0])

fig = plt.figure(figsize=(16,11))
ax = fig.add_subplot(111)
plt.scatter(score_diff['Final_Score'],score_diff['Rounded Viewers'], s=50,c ='orange', edgecolors='black', alpha=0.7)
plt.xlabel('Score Differences', fontsize=14)
plt.ylabel('Total Viewership',fontsize=14)
plt.title('Scatterplot of Score Differences vs. Total  Viewership',fontsize=18)
player_data


from datetime import datetime
game_data_viewers = game_data.merge(new_df, left_on = 'Game_ID', right_on = 'Game_ID',how='inner')
game_data_viewers = game_data_viewers[game_data_viewers['Location']=='H']
game_data_viewers['Game_ID'] = game_data_viewers['Game_ID'].astype('category')
game_data_viewers['Game_Date'] = pd.to_datetime(game_data_viewers['Game_Date'])
game_data_viewers['Weekday'] = game_data_viewers['Game_Date'].map(lambda x: x.isoweekday())
game_data_viewers['Month'] = game_data_viewers['Game_Date'].map(lambda x:x.month)
# only want certain weekday 

one_weekday = game_data_viewers[game_data_viewers['Weekday']==6].loc[:,['Game_Date','Rounded Viewers']]. \
  groupby('Game_Date').agg('sum').reset_index()
one_weekday['Num_of_games'] = one_weekday['Game_Date'].map(lambda x:len(game_data_viewers[game_data_viewers['Game_Date']==x]))
one_weekday
len(game_data_viewers[game_data_viewers['Game_Date']==x])
fig = plt.figure(1, figsize=(16, 7))
ax = fig.add_subplot(111)
game_data_viewers[game_data_viewers['Weekday']==1].plot(ax=ax)
game_data_viewers = game_data_viewers.loc[:,['Game_Date','Rounded Viewers']]
game_data_viewers = game_data_viewers.groupby('Game_Date').agg('sum')
game_data_viewers[game_data_viewers['Rounded Viewers'] == game_data_viewers['Rounded Viewers'].max()]
fig = plt.figure(1, figsize=(16, 7))
ax = fig.add_subplot(111)
game_data_viewers.loc[game_data_viewers.index < '2017-06-01'].plot(ax=ax)
fig = plt.figure(1, figsize=(16, 7))
ax = fig.add_subplot(111)
game_data_viewers.plot(ax=ax)
game_data_viewers.info()
player_data.columns
player_data
