

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

matches=pd.read_csv('../input/WorldCupMatches.csv')
#check null values in matches - 3720 records are null
matches.isnull().sum()
matches.dropna(inplace=True)
#add columns
matches['HomeWinGoals']=matches['Home Team Goals']-matches['Away Team Goals']
#define Home team win 
def home_winner(row):
    return ( row['HomeWinGoals']>0 or (row['Home Team Name'] in row['Win conditions']))*1

matches['HomeTeamWin'] = (matches.apply(home_winner,axis=1))
#define away team win
def away_winner(row):
    return ( row['HomeWinGoals']<0 or (row['Away Team Name'] in row['Win conditions']))*1

matches['AwayTeamWin'] = (matches.apply(away_winner,axis=1))
#combine stage Groups to one group

matches['Stage']=matches['Stage'].apply(lambda x: 'Group' if 'Group' in x else ('Match for third place' if 'hird' in x else x ) )
matches.Stage.value_counts()
#summarize for each team for their performance at different stages
def stagesummary(data, by, wintype):
    
    stagegroup=data.groupby([by,'Stage'])
    #wins for each team at different stage
    wins=stagegroup[[wintype]].sum()
    wins.reset_index(inplace=True)
    wins['Stage']=wins['Stage'].apply(lambda x:  x +'_win')
    wins=wins.pivot(index=by, columns='Stage',values=wintype).fillna(0)
    wins.reset_index(inplace=True)
    #plays for each team at different stage
    plays=stagegroup[['MatchID']].count()
    plays.reset_index(inplace=True)
    plays['Stage']=plays['Stage'].apply(lambda x: x+'_play')
    plays=plays.pivot(index=by, columns='Stage', values='MatchID').fillna(0)
    plays.reset_index(inplace=True)

    stagesummary=pd.merge(plays,wins, how='outer',on=by)
    stagesummary.rename(columns={by:'TeamName'},inplace=True)
    
    return stagesummary


stagesummary_home=stagesummary(matches,'Home Team Name','HomeTeamWin')
stagesummary_away=stagesummary(matches,'Away Team Name','AwayTeamWin')
stagesummary=pd.concat([stagesummary_home,stagesummary_away])
stagesum=stagesummary.groupby('TeamName').sum()
stagesum.reset_index(inplace=True)
stagesum.columns.T
def appendcolumns(col1,col2):
    teams=stagesum.loc[stagesum[col1]>0,['TeamName',col1]].rename(columns={col1:'stage'})
    teams['type']='play'
    winners=stagesum.loc[stagesum[col1]>0,['TeamName',col2]].rename(columns={col2:'stage'})
    winners['type']='win'
    stage=pd.concat([teams,winners])
    
    return stage
#import libraries for visulization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 


final=appendcolumns('Final_play','Final_win')
plt.figure(figsize=(10,8))
ax=sns.barplot(y='TeamName',x='stage',hue='type',data=final, orient='h',palette="BuGn_r")
plt.xlabel('Finals')
plt.show()
semi_finals=appendcolumns('Semi-finals_play','Semi-finals_win')
plt.figure(figsize=(10,8))
ax=sns.barplot(y='TeamName',x='stage',hue='type',data=semi_finals, orient='h',palette="BuGn_r")
plt.xlabel('Semi-finals')
plt.show()
quarter_finals=appendcolumns('Quarter-finals_play','Quarter-finals_win')
plt.figure(figsize=(12,8))
ax=sns.barplot(y='TeamName',x='stage',hue='type',data=quarter_finals, orient='h',palette="BuGn_r")
plt.xlabel('quarter-finals')
plt.show()
round_of_16=appendcolumns('Round of 16_play','Round of 16_win')
plt.figure(figsize=(16,8))
ax=sns.barplot(y='TeamName',x='stage',hue='type',data=round_of_16, orient='h',palette="BuGn_r")
plt.xlabel('round of 16')
plt.show()


