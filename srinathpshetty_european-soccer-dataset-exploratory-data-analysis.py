# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3 

import matplotlib.pyplot as plt

import seaborn as sns

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

from IPython.display import display, HTML

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Retrieve the connection object for sqlite3

conn=sqlite3.connect("/kaggle/input/soccer/database.sqlite")

tables=pd.read_sql("Select * from sqlite_master where type='table'",conn)

tables.head(10)
#Read data into DataFrames from all the available tables

PlayerAttribute_df=pd.read_sql("Select * from Player_Attributes",conn)

Player_df=pd.read_sql("Select * from Player",conn)

Match_df=pd.read_sql("Select * from Match",conn)

League_df=pd.read_sql("Select * from League",conn)

Country_df=pd.read_sql("Select * from Country",conn)

Team_df=pd.read_sql("Select * from Team",conn)

TeamAttributes_df=pd.read_sql("Select * from Team_Attributes",conn)
#Function to find the proportion of missing values

def missing_values(input_df,null_percent=0):

    output_df=pd.DataFrame({'missing_count':[],'missing_prop':[]})

    nullcount_df=input_df.isna().sum()

    output_df['missing_count']=nullcount_df.iloc[0:]

    output_df['missing_prop']=output_df['missing_count']/len(input_df.index)*100

    output_df.index=nullcount_df.index

    if null_percent>0:

        return output_df[output_df['missing_prop']>=null_percent]

    else:

        return output_df
missing_values(Match_df,20)
#Merge Match_df with Country_df

MatchMerged_df=Match_df.merge(Country_df,left_on='country_id',right_on='id',how='left',suffixes=('_match','_country'))

MatchMerged_df.rename(columns={'name':'country'},inplace=True)



#Merge with League info

MatchMerged_df=MatchMerged_df.merge(League_df,left_on='league_id',right_on='id',how='left',suffixes=('_match','_league'))

MatchMerged_df.rename(columns={'id':'id_league','name':'league'},inplace=True)



#Merge with Team Info

MatchMerged_df=MatchMerged_df.merge(Team_df[['team_api_id','team_long_name']],left_on='home_team_api_id',right_on='team_api_id',how='left',suffixes=('_match','_team'))

MatchMerged_df.rename(columns={'team_api_id':'team_api_id_home','team_long_name':'home_team'},inplace=True)

MatchMerged_df=MatchMerged_df.merge(Team_df[['team_api_id','team_long_name']],left_on='away_team_api_id',right_on='team_api_id',how='left',suffixes=('_match','_away'))

MatchMerged_df.rename(columns={'team_api_id':'team_api_id_home','team_long_name':'away_team'},inplace=True)



#Reformat Date Column

MatchMerged_df['date']=pd.to_datetime(MatchMerged_df['date'])



#Create New Column as 'total_no_goals'= 'home_team_goal'+'away_team_goal'

MatchMerged_df['total_no_goals']=MatchMerged_df['home_team_goal']+MatchMerged_df['away_team_goal']



#DataFrame(in_use).loc['condition to be satisfied','new_column_name']='value, if condition is True'



#Create new column 'result' that identifies the result of the match (HTW: Home Team Win, ATW: Away team Win, D: Draw)

MatchMerged_df.loc[MatchMerged_df['home_team_goal']>MatchMerged_df['away_team_goal'],'result']='HTW'

MatchMerged_df.loc[MatchMerged_df['away_team_goal']>MatchMerged_df['home_team_goal'],'result']='ATW'

MatchMerged_df.loc[MatchMerged_df['home_team_goal']==MatchMerged_df['away_team_goal'],'result']='D'



#Create new column 'winning_team' that identifies the winning team

MatchMerged_df.loc[MatchMerged_df['home_team_goal']>MatchMerged_df['away_team_goal'],'winning_team']=MatchMerged_df['home_team']

MatchMerged_df.loc[MatchMerged_df['away_team_goal']>MatchMerged_df['home_team_goal'],'winning_team']=MatchMerged_df['away_team']

MatchMerged_df.loc[MatchMerged_df['home_team_goal']==MatchMerged_df['away_team_goal'],'winning_team']=np.nan



#Create new column 'losing_team' that identifies the losing team

MatchMerged_df.loc[MatchMerged_df['home_team_goal']>MatchMerged_df['away_team_goal'],'losing_team']=MatchMerged_df['away_team']

MatchMerged_df.loc[MatchMerged_df['away_team_goal']>MatchMerged_df['home_team_goal'],'losing_team']=MatchMerged_df['home_team']

MatchMerged_df.loc[MatchMerged_df['home_team_goal']==MatchMerged_df['away_team_goal'],'losing_team']=np.nan



MatchMerged_df.head()
#Create new dataframe with only required columns

NewMatch_df=MatchMerged_df[['season','stage','date','country','league','home_team','away_team','home_team_goal','away_team_goal','total_no_goals',

                           'winning_team','losing_team','result']]

NewMatch_df.head()
#Create dataframe with each column representing the proportion information for each category

Prop_df=pd.DataFrame({

    'Home Team Wins':[len(NewMatch_df[NewMatch_df['result']=='HTW'])/len(NewMatch_df)*100],

    'Away Team Wins':[len((NewMatch_df[NewMatch_df['result']=='ATW']))/len(NewMatch_df)*100],

    'Draw':[len(NewMatch_df[NewMatch_df['result']=='D'])/len(NewMatch_df)*100]

})



plt.figure(figsize=(5,5))

patches,text,autotext=plt.pie(Prop_df.iloc[0],labels=Prop_df.columns,startangle=0,autopct="%1.2f%%",

                              explode=(0.05,0.05,0.05),radius=2,colors=sns.color_palette('pastel',3),shadow=True,

                              wedgeprops={'edgecolor':'k'})



#Set font size for labels

for  i in text:

    i.set_fontsize(15)

for j in autotext:

    j.set_fontsize(15)



plt.show()
#Split the match dataframe to get home team & away team statistics

home_team_df=MatchMerged_df[['season','league','home_team_api_id','home_team','away_team','home_team_goal','away_team_goal']]

away_team_df=MatchMerged_df[['season','league','away_team_api_id','away_team','home_team','away_team_goal','home_team_goal']]



home_team_df.rename(columns={'home_team_api_id':'team_api_id',

                 'home_team':'team',

                 'away_team':'opp_team',

                 'home_team_goal':'team_goal',

                 'away_team_goal':'opp_team_goal'},inplace=True)

away_team_df.rename(columns={'away_team_api_id':'team_api_id',

                 'away_team':'team',

                 'home_team':'opp_team',

                 'away_team_goal':'team_goal',

                 'home_team_goal':'opp_team_goal'},inplace=True)



TeamPoints_df=pd.concat([home_team_df,away_team_df],axis=0,ignore_index=True)



#Create 'Goal Difference' column

TeamPoints_df['goal_diff']=TeamPoints_df['team_goal']-TeamPoints_df['opp_team_goal']



#Create 'team_points' column for each team with following points system, (3: Winning Team, 1: Draw)

TeamPoints_df.loc[TeamPoints_df['team_goal']>TeamPoints_df['opp_team_goal'],'team_points']=3

TeamPoints_df.loc[TeamPoints_df['team_goal']==TeamPoints_df['opp_team_goal'],'team_points']=1



#Create 'Wins', 'Draws' & 'Losses' column 

TeamPoints_df.loc[TeamPoints_df['team_goal']>TeamPoints_df['opp_team_goal'],'wins']=1

TeamPoints_df.loc[TeamPoints_df['team_goal']<TeamPoints_df['opp_team_goal'],'losses']=1

TeamPoints_df.loc[TeamPoints_df['team_goal']==TeamPoints_df['opp_team_goal'],'draw']=1



#Capture matches played

TeamPoints_df['matches_played']=1



#Rearranging the tables to match actual score tables

TeamPoints_df.reindex(columns=['season', 'league', 'team','team_api_id', 'opp_team', 'matches_played',

'wins','draw','losses','team_goal','opp_team_goal','goal_diff','team_points'],copy=False)



#Generate Score table, by grouping the dataframe on 'league','season','team','team_api_id'. Also, let's sort it based on the following parameters in the below mentioned order

ScoreTable_df=TeamPoints_df.groupby(['league','season','team','team_api_id']).sum().sort_values(['league','season','team_points','goal_diff','team_goal'],ascending=[True,True,False,False,False]).reset_index()

ScoreTable_df.head()
#Dropdown for League information

League=widgets.Dropdown(

    options=[i for i in ScoreTable_df['league'].unique()],

    description='League:',

    disabled=False,

)



#Dropdown for Season information

Season=widgets.Dropdown(

    options=[i for i in ScoreTable_df['season'].unique()],

    description='Season:',

    disabled=False,

)



#Range Slider for Rank information

Rank_Range=widgets.IntRangeSlider(

    value=(1,ScoreTable_df[(ScoreTable_df['league']==League.value)&(ScoreTable_df['season']==Season.value)]['team'].nunique()),

    min=1,

    max=ScoreTable_df[(ScoreTable_df['league']==League.value)&(ScoreTable_df['season']==Season.value)]['team'].nunique(),

    step=1,

    description='Rank:',

    disabled=False,

    continuous_update=False,

    orientation='horizontal',

    readout=True,

    readout_format='d',

)



def update_range(*args):

    Rank_Range.value=(1,ScoreTable_df[(ScoreTable_df['league']==League.value)&(ScoreTable_df['season']==Season.value)]['team'].nunique())

    Rank_Range.max=ScoreTable_df[(ScoreTable_df['league']==League.value)&(ScoreTable_df['season']==Season.value)]['team'].nunique()



#Observe changes in League & Season Dropdowns

League.observe(update_range,'value')

Season.observe(update_range,'value')



def team_results(league_val,season_val,range_val):

    MatchTable_df=ScoreTable_df[(ScoreTable_df['league']==league_val)&(ScoreTable_df['season']==season_val)].copy()

    MatchTable_df.rename(columns={'team':'Club','matches_played':'MP','wins':'W','draw':'D','losses':'L',

                                  'team_goal':'GF','opp_team_goal':'GA','goal_diff':'GD','team_points':'Pts'},inplace=True)

    MatchTable_df=MatchTable_df[['Club','MP','W','D','L','GF','GA','GD','Pts']].iloc[range_val[0]-1:range_val[1]]

    MatchTable_df.set_index(np.arange(1,len(MatchTable_df.index)+1),inplace=True)

    #We convert the DF to HTML using to_html and styple it using the classes of bootstrap

    display(HTML(MatchTable_df.to_html(classes='table table-striped table-hover table')))



Team_result=interact(team_results,league_val=League,season_val=Season,range_val=Rank_Range)
#Calculate Average Goals per League

LeagueAvgGoal_df=NewMatch_df.groupby(['league']).mean().sort_values(['total_no_goals'],ascending=False).reset_index()

LeagueAvgGoal_df.head()
#Plot average goals scored per league

sns.set_style('white')

a=sns.catplot(kind='bar',x='total_no_goals',y='league',data=LeagueAvgGoal_df,ci='std',edgecolor='k')

a.fig.set_size_inches(12,8)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('Average Goals Scored',fontsize=15)

plt.ylabel('League',fontsize=15)

plt.xlim(0,4)

plt.title('Avg. Goals scored per League',fontsize=20)

for i in range(len(LeagueAvgGoal_df)):

    plt.text(LeagueAvgGoal_df['total_no_goals'].iloc[i]+0.15,i+0.15,round(LeagueAvgGoal_df['total_no_goals'].iloc[i],2),fontsize=12)    
#Find the number of Home Team Wins, Away Team Wins & Draws, by pivoting on League & the match results and counting the number of matches played

#Also, sum the columns to find the total number of matches played in the League

LeagueProp=MatchMerged_df.pivot_table(index='league',columns='result',values='match_api_id',aggfunc='count')

LeagueProp['sum']=LeagueProp.sum(axis=1)

LeagueProp.head()
#Divide the match results columns with the sum column to find their proportions

LeagueProp=LeagueProp[['HTW','ATW','D']].divide(LeagueProp['sum'],axis=0).multiply(100)

LeagueProp.head()
ax=LeagueProp.plot.barh(stacked=True,figsize=(10,8),width=0.75,color=sns.color_palette("hls", 8),edgecolor='w',alpha=0.8)

ax.legend(['Home Team','Away Team','Draw'],bbox_to_anchor=(1.2,1),loc='upper right')

plt.title('Predictability of an outcome in a League Match',fontsize=20)

plt.xlabel('Proportion of Home Team Wins vs Away Team Wins vs Draws',fontsize=15)

plt.ylabel('Leagues',fontsize=15)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

for  i,j in enumerate(LeagueProp.index):

    plt.text(LeagueProp.loc[j,'HTW']/2,i,str(round(LeagueProp.loc[j,'HTW'],2))+' %',fontsize=12)

    plt.text(LeagueProp.loc[j,'HTW']+LeagueProp.loc[j,'ATW']/2,i,str(round(LeagueProp.loc[j,'ATW'],2))+' %',fontsize=12)

    plt.text(LeagueProp.loc[j,'HTW']+LeagueProp.loc[j,'ATW']+LeagueProp.loc[j,'D']/2,i,str(round(LeagueProp.loc[j,'D'],2))+'%',fontsize=12)
#Group the Match statistics on the basis of the Teams, and create new columns to calculate winning & losing proportion

Consolidated_df=ScoreTable_df.groupby(['team']).sum()

Consolidated_df['win_prop']=round(Consolidated_df['wins']/Consolidated_df['matches_played']*100,2)

Consolidated_df['loss_prop']=round(Consolidated_df['losses']/Consolidated_df['matches_played']*100,2)



#Sort based on columns win_prop & loss_prop to find the Most Winning & Losing teams respectively

MostWinningTeam_df=Consolidated_df.sort_values('win_prop',ascending=True).tail(10)['win_prop']

MostLosingTeam_df=Consolidated_df.sort_values('loss_prop',ascending=True).tail(10)['loss_prop']



plt.figure(figsize=(14,8))

plt.subplot(1,2,1)

MostWinningTeam_df.plot.barh(color=sns.color_palette('Greens', 10),edgecolor='k')

plt.title('Most Winning Team in History',fontsize=20)

plt.xlim(0,100)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('% of Wins',fontsize=15)

plt.ylabel('Teams',fontsize=15)

for i,j in enumerate(MostWinningTeam_df.index):

    plt.text(MostWinningTeam_df.loc[j]+1.5,i,str(MostWinningTeam_df.loc[j])+'%',style='italic',fontsize=12)    



plt.subplot(1,2,2)

MostLosingTeam_df.plot.barh(color=sns.color_palette('Reds', 10),edgecolor='k')

plt.title('Most Losing Team in History',fontsize=20)

plt.xlim(0,100)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel('% of Losses',fontsize=15)

plt.ylabel('')

for i,j in enumerate(MostLosingTeam_df.index):

    plt.text(MostLosingTeam_df.loc[j]+1.5,i,str(MostLosingTeam_df.loc[j])+'%',style='italic',fontsize=12)



plt.subplots_adjust(wspace=0.5)
Top3Team_df=TeamPoints_df.groupby(['league','team']).count()

Top3Team_df['win_percent']=Top3Team_df['wins']/Top3Team_df['matches_played']*100

Top3Team_df=Top3Team_df['win_percent'].groupby(level=0,group_keys=False)

Top3Team_df=Top3Team_df.nlargest(3)

Top3Team_df.head(9)
#Transform the above result into a DataFrame

rank1_team=[]

rank2_team=[]

rank3_team=[]

rank1_value=[]

rank2_value=[]

rank3_value=[]

league_index=[]



for i in League_df['name']:

    league_index.append(i)

    rank1_team.append(Top3Team_df[i].index[0])

    rank2_team.append(Top3Team_df[i].index[1])

    rank3_team.append(Top3Team_df[i].index[2])

    rank1_value.append(Top3Team_df[i].values[0])

    rank2_value.append(Top3Team_df[i].values[1])

    rank3_value.append(Top3Team_df[i].values[2])



TopLeagueTeam_df=pd.DataFrame({'rank1_team':rank1_team,'rank2_team':rank2_team,'rank3_team':rank3_team,

                               'rank1_value':rank1_value,'rank2_value':rank2_value,'rank3_value':rank3_value},

                                index=league_index)
fig,ax=plt.subplots(figsize=(12,12))

pos=list(range(0,len(TopLeagueTeam_df.index)))

w=0.25

a=0.75

plt.barh(pos,TopLeagueTeam_df['rank3_value'],w,color=sns.color_palette('hls', 12),alpha=a-0.5,edgecolor='k')

plt.barh([i+w for i in pos],TopLeagueTeam_df['rank2_value'],w,color=sns.color_palette('hls', 12),alpha=a-0.25,edgecolor='k')

plt.barh([i+w*2 for i in pos],TopLeagueTeam_df['rank1_value'],w,color=sns.color_palette('hls', 12),alpha=a,edgecolor='k')

plt.ylabel('LEAGUES',fontsize=15)

plt.xlabel('% WINS OF ALL MATCHES PLAYED',fontsize=15)

plt.xlim(0,100)

plt.yticks([i+0.25 for i in pos],TopLeagueTeam_df.index,fontsize=12)

plt.title('Top Teams in each League',fontsize=20)

plt.grid()

for i,j in enumerate(TopLeagueTeam_df.index):

   plt.text(TopLeagueTeam_df['rank3_value'][i]+1,i-0.05,TopLeagueTeam_df['rank3_team'][i],fontsize=10,)

   plt.text(TopLeagueTeam_df['rank2_value'][i]+1,i+0.2,TopLeagueTeam_df['rank2_team'][i],fontsize=10)

   plt.text(TopLeagueTeam_df['rank1_value'][i]+1,i+0.45,TopLeagueTeam_df['rank1_team'][i],fontsize=10,fontweight='bold')
Team_df.info()
TeamAttributes_df.info()
#Look at the actual data for 'buildUpPlayDribbling','buildUpPlayDribblingClass'

bupd_df=TeamAttributes_df[['buildUpPlayDribbling','buildUpPlayDribblingClass']]

bupd_df.head(10)
bupd_df[bupd_df['buildUpPlayDribbling'].isna()]['buildUpPlayDribblingClass'].unique()
bupd_df[bupd_df['buildUpPlayDribblingClass']=='Little'].describe()  
#Replace Null values in TeamAttributes_df.buildUpPlayDribbling with 30

TeamAttributes_df['buildUpPlayDribbling'][TeamAttributes_df['buildUpPlayDribbling'].isna()]=30

TeamAttributes_df[['buildUpPlayDribbling','buildUpPlayDribblingClass']]
#Group Match statistics on the basis of team and create new column 'avg_team_pts' that represent the average point gained by team

EffTeam_df=ScoreTable_df.groupby(['team_api_id','team']).sum().reset_index()

EffTeam_df['avg_team_pts']=round(EffTeam_df['team_points']/(EffTeam_df['matches_played']),2)

EffTeam_df.head()
#Let's understand the distribution of the average team points

EffTeam_df.describe(percentiles=[0.1,0.25,0.50,0.75,0.90])
TeamAttributes_df.head()
#Group the Team Attributes on basis of the team and merge it with Team & Match info

SummTeamAtt_df=TeamAttributes_df.groupby(['team_api_id']).mean()

MergeTeamAtt_df=Team_df.merge(SummTeamAtt_df,on='team_api_id')

MergeTeamAtt_df=EffTeam_df.merge(MergeTeamAtt_df,on='team_api_id')



#Filter out the Top10 teams from Rest of the Teams

Top10Teams_df=MergeTeamAtt_df[MergeTeamAtt_df['avg_team_pts']>1.755].mean()

RestOfTeams_df=MergeTeamAtt_df[MergeTeamAtt_df['avg_team_pts']<=1.755].mean()



#Make a list to identify Top 10 teams and the rest of the teams

Top10Teams_id=EffTeam_df[EffTeam_df['avg_team_pts']>1.755]['team_api_id'].values

RestOfTeams_id=EffTeam_df[EffTeam_df['avg_team_pts']<=1.755]['team_api_id'].values
#Plot radar graph to compare the Team Attributes

from math import pi



Top10TeamsVal=list(Top10Teams_df[14:].values)

Top10TeamsVal.append(Top10TeamsVal[0])

RestOfTeamsVal=list(RestOfTeams_df[14:].values)

RestOfTeamsVal.append(RestOfTeamsVal[0])

categories=Top10Teams_df.index[14:]

N=len(categories)



angles=[2*pi*i/N for i in range(N)]

angles.append(angles[0])



fig=plt.figure(figsize=(10,10))

ax=fig.add_subplot(111,polar=True)



plt.title("Team Attribute Comparison (Top 10% vs Rest)",fontsize=20)



plt.xticks(angles[:-1],categories)





ax.plot(angles,Top10TeamsVal,'o-',linewidth=2,label='Top 10 Teams')

ax.fill(angles,Top10TeamsVal,alpha=0.25)



ax.plot(angles,RestOfTeamsVal,'o-',linewidth=2,label='Rest of Team')

ax.fill(angles,RestOfTeamsVal,alpha=0.25)



ax.legend()



plt.show()
#First we need the Home & Away team player details for each match played

home_team_player=pd.melt(MatchMerged_df,id_vars=['season','date','match_api_id','home_team_api_id'],var_name='player',value_name='player_api_id',

                         value_vars=['home_player_1','home_player_2','home_player_3','home_player_4',

                                    'home_player_5','home_player_6','home_player_7','home_player_8',

                                    'home_player_9','home_player_10','home_player_11'])

away_team_player=pd.melt(MatchMerged_df,id_vars=['season','date','match_api_id','away_team_api_id'],var_name='player',value_name='player_api_id',

                         value_vars=['away_player_1','away_player_2','away_player_3','away_player_4',

                                    'away_player_5','away_player_6','away_player_7','away_player_8',

                                    'away_player_9','away_player_10','away_player_11'])



home_team_player.rename(columns={'home_team_api_id':'team_api_id'},inplace=True)

away_team_player.rename(columns={'away_team_api_id':'team_api_id'},inplace=True)



players_per_season=pd.concat([home_team_player,away_team_player],ignore_index=True)



players_per_season.tail()
#Convert date column to datetime

PlayerAttribute_df['date']=pd.to_datetime(PlayerAttribute_df['date'])



#Create new column 'season', which will be used to merge with players_per_season, 

#If the month in which the attribute was recorded is greater than 6(June), Then season=current year/next year Else season=previous year/current year

PlayerAttribute_df['season']=np.where(PlayerAttribute_df['date'].dt.month>6,(PlayerAttribute_df['date'].dt.year).astype(str)+'/'+(PlayerAttribute_df['date'].dt.year+1).astype(str),

                                                                            (PlayerAttribute_df['date'].dt.year-1).astype(str)+'/'+(PlayerAttribute_df['date'].dt.year).astype(str))
#Average the attributes of a player in a season

PlayerAttribute_df=PlayerAttribute_df.groupby(['player_api_id','season']).mean().reset_index()

PlayerAttribute_df.head()
#Merge information from players season information with player attributes & player information

player_per_season_att_df=pd.merge(players_per_season,PlayerAttribute_df,on=['season','player_api_id'],how='left')

player_per_season_att_df=pd.merge(player_per_season_att_df,Player_df,on=['player_api_id'],how='left')
player_per_season_att_df.tail(5)
#Create a column 'age'

player_per_season_att_df['birthday']=pd.to_datetime(player_per_season_att_df['birthday'])

player_per_season_att_df['age']=player_per_season_att_df['date'].dt.year-player_per_season_att_df['birthday'].dt.year
player_per_season_att_df.info()
#Create dataframe to capture top teams performance vs rest of the teams

player_att_comp_df=pd.concat([player_per_season_att_df[player_per_season_att_df.team_api_id.isin(list(Top10Teams_id))].mean(),

                              player_per_season_att_df.mean()],axis=1)

player_att_comp_df.rename(columns={0:'top10',1:'rest'},inplace=True)



#create a column 'percent_diff' that captures the percentage difference of the player attributes between top team and rest of the team

player_att_comp_df['percent_diff']=(player_att_comp_df['top10']/player_att_comp_df['rest']-1)*100



player_att_comp_df.drop(index=['match_api_id','team_api_id','player_api_id','id_x','player_fifa_api_id_x','id_y','player_fifa_api_id_y'],inplace=True)



player_att_comp_df.sort_values(by=['percent_diff'],ascending=[True],inplace=True)
player_att_comp_df
plt.figure(figsize=(12,20))

plt.barh(player_att_comp_df.index,player_att_comp_df['percent_diff'],color=sns.color_palette('coolwarm',40),edgecolor='k',alpha=0.6)

plt.title("Player Attribute Comparison (Top 10% vs Rest)",fontsize=25)

plt.ylabel('Player Attributes',fontsize=20)

plt.xlabel('% Difference of Mean of Top 10% vs Rest',fontsize=20)

plt.xticks(size=13)

plt.yticks(size=13)

plt.xlim(-10,20)

plt.grid()



plt.show()