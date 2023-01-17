import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Reading the match data from the csv file
match = pd.read_csv(r'/kaggle/input/ipldata/matches.csv')
match.head(2)
# Reading the data for each ball 
ball= pd.read_csv(r'/kaggle/input/ipldata/deliveries.csv')
ball.head(2)

match.info()
match[match.isna()['winner']]
ball.info()
match.duplicated().sum()
ball.duplicated().sum()
ball[ball.duplicated()].head()
ball.query('match_id==221 & inning==1 & over==4 & ball==1')
ball.drop_duplicates(inplace= True)
# Replacing Franchise names
ball.replace({'Sunrisers Hyderabad':'Hyderabad (Sunriser/Chargers)','Deccan Chargers':'Hyderabad (Sunriser/Chargers)',\
'Rising Pune Supergiants':'Pune (Supergiant/ Warriors)','Delhi Daredevils':'Delhi (Capitals/ Daredevils)',\
'Delhi Capitals':'Delhi (Capitals/ Daredevils)','Pune Warriors':'Pune (Supergiant/ Warriors)',
'Rising Pune Supergiant':'Pune (Supergiant/ Warriors)'}, inplace= True)
ball.batting_team.unique()
match.replace({'Sunrisers Hyderabad':'Hyderabad (Sunriser/Chargers)','Deccan Chargers':'Hyderabad (Sunriser/Chargers)',\
'Rising Pune Supergiants':'Pune (Supergiant/ Warriors)','Delhi Daredevils':'Delhi (Capitals/ Daredevils)',\
'Delhi Capitals':'Delhi (Capitals/ Daredevils)','Pune Warriors':'Pune (Supergiant/ Warriors)',
'Rising Pune Supergiant':'Pune (Supergiant/ Warriors)'}, inplace= True)
match.team2.nunique()
# Creating a new df summarizing total runs and wickets that fell in each innings

summary= ball.groupby(by=['match_id','inning','batting_team']).sum()['total_runs'].reset_index()
temp = ball.groupby(by=['match_id','inning','batting_team']).count()['player_dismissed'].reset_index()

summary= pd.merge(summary, temp,how='outer', on= ['match_id','inning', 'batting_team'])
# Creating a new column combining the over and the ball bowled
ball['over_ball']= ball['over']+ball['ball']/10
# Merging the total balls used in the innings with the summary data
temp= ball.groupby(by=['match_id','inning','batting_team']).max(numeric_only=True)['over_ball'].reset_index()

summary= pd.merge(summary, temp,how='outer', on= ['match_id','inning', 'batting_team'])
summary.head(2)
# Changing the matrix into a long format
s1 = summary.melt(id_vars=['match_id','inning',]).query('variable=="total_runs" & inning==1')[['match_id','value']]
s2 =summary.melt(id_vars=['match_id','inning',]).query('variable=="total_runs" & inning==2')[['match_id','value']]
temp= pd.merge(s1, s2, on='match_id', how='inner')

s1= summary.melt(id_vars=['match_id','inning',]).query('variable=="over_ball" & inning==1')[['match_id','value']]
s2=summary.melt(id_vars=['match_id','inning',]).query('variable=="over_ball" & inning==2')[['match_id','value']]
temp2= pd.merge(s1,s2,on='match_id', how='inner')

s1= summary.melt(id_vars=['match_id','inning',]).query('variable=="player_dismissed" & inning==1')[['match_id','value']]
s2=summary.melt(id_vars=['match_id','inning',]).query('variable=="player_dismissed" & inning==2')[['match_id','value']]
temp3 = pd.merge(s1,s2,on='match_id', how='inner')

summary = pd.merge(pd.merge(temp,temp2, on= 'match_id', how='inner'), temp3, on='match_id', how='inner')

summary.rename(columns={'value_x_x':'runs_1', 'value_y_x':'runs_2','value_x_y':'overs_1','value_y_y':'overs_2',\
    'value_x':'wickets_1', 'value_y':'wickets_2'}, inplace= True)
summary.head()
# Merging the summary data with match info data
match= pd.merge(match, summary, right_on= 'match_id', left_on= 'id', how='left')
match.drop(columns=['umpire1', 'umpire2', 'umpire3'], inplace= True)
match.head(2)
# Total runs scored by each batsman
batsman= ball.groupby(by='batsman').sum()['batsman_runs'].reset_index()

#Total balls played by each batsman
temp = ball.groupby(by='batsman').count()['ball']
batsman= pd.merge(batsman, temp, how='outer',on='batsman')

# Total 4s hit
temp = ball.query('batsman_runs==4').groupby(by='batsman').count()['batsman_runs'].reset_index()
batsman = pd.merge(batsman, temp, how='outer',on='batsman')

#Total 6s hit
temp= ball.query('batsman_runs==6').groupby(by='batsman').count()['batsman_runs'].reset_index()
batsman = pd.merge(batsman, temp, how='outer',on='batsman')

batsman.rename(columns={'batsman_runs_x':'total_runs','batsman_runs_y':'4s', 'batsman_runs':'6s'}, inplace= True)
batsman.head(3)
# Calculating the strike rate
batsman['strike_rate']= (batsman.total_runs/ batsman.ball)*100
batsman.head(2)
# Calculating percent of the runs scored by boundries
batsman['boundary_percent']= (batsman['4s']*4+ batsman['6s']*6)/ batsman.total_runs
batsman.head(1)
# Calculating the runs scored by openers in each match
#calculating the runs scored by each batsman in each match
a= ball.groupby(by=['match_id','batsman']).sum()['batsman_runs']
b= ball.groupby(by=['match_id','batsman']).count()['total_runs']
opening = pd.DataFrame(data=[a,b]).transpose().reset_index()

# finding which batsmen opened
opening2= ball.query('over==1 & ball==1')[['match_id','batsman','non_striker']].melt(id_vars='match_id')
opening = pd.merge(opening, opening2, how='right',left_on =['match_id','batsman'], right_on=['match_id','value'])

#renaming columns
opening.rename(columns={'total_runs':'opening_balls', 'batsman_runs':'opening_runs'}, inplace= True)
#dropping redundant columns
opening.drop(columns=['variable','value'], inplace= True)
opening.head()
# Aggregating the runs scored by openers
opening= opening.groupby(by='batsman').sum()[['opening_runs','opening_balls']].reset_index()

batsman= pd.merge(batsman, opening, on='batsman', how='left')
batsman.head(3)

ball.dismissal_kind.value_counts()
ball.head(2)
# Total balls bowled and total runs given by the bowler
bowler= ball.groupby(by='bowler').count()['ball'].reset_index()
temp= ball.groupby(by='bowler').sum()['total_runs']

bowler= pd.merge(bowler, temp, on='bowler', how='outer')
# Filtering the runouts/ retired hurts
temp = ball.query('dismissal_kind!="obstructing the field" & dismissal_kind !="retired hurt" & dismissal_kind != "run out"')

# Filtering the wickets taken
temp = temp[~temp.isna().dismissal_kind][['bowler','dismissal_kind']]
temp= temp.groupby(by=['bowler']).count().reset_index()

bowler= pd.merge(bowler, temp, on='bowler', how='outer')
# Renaming the column to reflect the wickets
bowler.rename(columns={'dismissal_kind':'wickets_taken'}, inplace= True)
# Calculating the total dot balls
temp = ball.query('total_runs==0').groupby(by='bowler').count()['ball'].reset_index()
bowler= pd.merge(bowler, temp, on='bowler', how='outer')
# Renaming the column to reflect the wickets
bowler.rename(columns={'ball_x':'balls_bowled', 'ball_y':'dot_balls'}, inplace= True)
bowler.head(1)
# Calculating the number of bowled and LBWs taken by the bowler
temp =ball.query('dismissal_kind=="bowled" | dismissal_kind =="lbw" ').groupby(by='bowler').count()['dismissal_kind'].reset_index()
bowler= pd.merge(bowler, temp, on='bowler', how='outer')
bowler.head(3)
# Renaming the column to reflect the wickets
bowler.rename(columns={'dismissal_kind':'bowl_or_lbw'}, inplace= True)
bowler.head(1)
# Calculating the bowler economy
bowler['economy']= bowler.total_runs/ (bowler.balls_bowled/6)
# Calculating dot ball percent
bowler['dot_ball_percent']= bowler.dot_balls/bowler.balls_bowled
# Calculating the runs scored in last 3 overs (death over) to compare the bowlers performance
a= ball.query('over>=18').groupby('bowler').sum()['total_runs']
b= ball.query('over>=18').groupby('bowler').count()['batsman_runs']
temp =pd.DataFrame(data=[a,b] ).transpose().reset_index()
temp.rename(columns={'total_runs':'death_runs','batsman_runs':'death_balls' }, inplace= True)

bowler= pd.merge(bowler, temp, on='bowler', how='outer')
bowler.head(3)
# Calculating bowling economy in death over
bowler['death_econ']= bowler.death_runs/ (bowler.death_balls/6)
# Calculating the percentage of bowled and LBWs by the bowler among the wickets they took
bowler['bowl_lbw_percent']= bowler.bowl_or_lbw/ bowler.wickets_taken
bowler.head(1)
# Total innings played for each over bowled
overs_played= ball[['batting_team','over','match_id']].drop_duplicates().groupby(by=['batting_team','over']).count()['match_id'].reset_index()
# Total runs scored in each over
over_runs= ball.groupby(by=['batting_team','over']).sum()['total_runs'].reset_index()
over_runs= pd.merge(over_runs, overs_played, on=['batting_team','over'], how='outer')

over_runs.rename(columns={'match_id':'matches'}, inplace= True)
# Calculating average runs scored in each over
over_runs['avg']= over_runs.total_runs/ over_runs.matches
over_runs.head()
# Changing the data type for the match columns to float values
float_list=['runs_1', 'runs_2','overs_1', 'overs_2','wickets_1', 'wickets_2' ]
for col in float_list:
    match.loc[:,col] = match[col].astype(float)
match.info()
match.replace({'Bengaluru':'Bangalore'}, inplace= True)
match['batting_first_won']= match.winner== match.team1
# Plotting average runs scored in each over in IPL
series= over_runs.groupby(by='over').mean()['avg']
sns.set_style('darkgrid')

plt.figure(figsize=(10,6))
sns.barplot(x=series.index, y=series, color='tab:blue', )
plt.axhline(over_runs.avg.mean(), color='r', label= 'Avg runs scored total')
plt.title('Average runs scored in each over')
plt.ylabel('Average runs scored')
plt.xlabel('Over number')
plt.yticks(np.arange(0,11))
plt.legend()
plt.show()
color_code= {'Chennai Super Kings':'y','Delhi (Capitals/ Daredevils)':'b','Hyderabad (Sunriser/Chargers)':'tab:orange',\
  'Kings XI Punjab':'r', 'Kolkata Knight Riders':'k','Mumbai Indians':'tab:blue','Rajasthan Royals':'tab:pink',\
     'Royal Challengers Bangalore':'tab:red'}
# Average runs scored per over team wise
sns.set_style('darkgrid')
plt.figure(figsize=(10,6))

temp= over_runs.query('batting_team !="Kochi Tuskers Kerala" & batting_team != "Pune (Supergiant/ Warriors)" & \
batting_team !="Gujarat Lions"')
sns.lineplot(x='over', y='avg', data=temp, hue='batting_team', palette=color_code)
plt.xticks(np.arange(1,21))
plt.legend(loc=(1.05,0.3))
plt.title('Average runs scored per over (team wise)')
plt.xlabel('Over')
plt.ylabel('Average runs scored')
plt.show()
# Grouping the wins for teams batting first and second
temp = match.groupby(by=['winner','team1']).count()['id'].reset_index().query('winner==team1')[['winner','id']]
temp2 = match.groupby(by=['winner','team2']).count()['id'].reset_index().query('winner==team2')[['winner','id']]
temp3= pd.DataFrame([match.team1.value_counts(),match.team2.value_counts()],).transpose().reset_index()

team_stat= pd.merge(temp, temp2, on='winner', how='inner')
team_stat= pd.merge(team_stat, temp3, left_on='winner',right_on='index', how='inner')

team_stat.rename(columns={'id_x':'batting_first','id_y':'batting_second'}, inplace= True)
team_stat.drop(columns=['index'], inplace= True)
# finding win percent inning wise
team_stat['win_percent_first']= team_stat['batting_first']/ team_stat['team1']
team_stat['win_percent_second']= team_stat['batting_second']/ team_stat['team2']
# Aggregating the win percent
team_stat['win_percent']=(team_stat.batting_first+ team_stat.batting_second)/ (team_stat.team1+ team_stat.team2)
# Sorting the values
team_stat.sort_values(by='win_percent', inplace=True, ascending=False)
plt.figure(figsize=(8,6))
sns.set_style('darkgrid')

sns.lineplot(x= 'winner', y= 'win_percent_first',data=team_stat, color='g',label='Batting First', sort= False, marker='o' )
sns.lineplot(x= 'winner', y= 'win_percent_second', data=team_stat, color='y', label= 'Fielding First',sort=False,marker='o')
sns.barplot(x='winner', y= 'win_percent',data=team_stat,  color='tab:blue', label='Average winning rate')

plt.xticks(rotation=90)
plt.xlabel('IPL Team')
plt.ylabel('Winning Percent')
plt.title('Winning percent for each team')
plt.legend()
plt.show()
team_stat
# Finding matches where Team 1 won by less than 6 runs or Team 2 won in last 3 balls
cliff= match.query('(win_by_runs <6 & win_by_runs >0) | (overs_2 > 20.3 & win_by_wickets >0)')

# Grouping the wins for teams batting first and second
temp = cliff.groupby(by=['winner','team1']).count()['id'].reset_index().query('winner==team1')[['winner','id']]
temp2 = cliff.groupby(by=['winner','team2']).count()['id'].reset_index().query('winner==team2')[['winner','id']]
# Grouping the wins for teams batting first and second
temp = cliff.groupby(by=['winner','team1']).count()['id'].reset_index().query('winner==team1')[['winner','id']]
temp2 = cliff.groupby(by=['winner','team2']).count()['id'].reset_index().query('winner==team2')[['winner','id']]
temp3= pd.DataFrame([cliff.team1.value_counts(),cliff.team2.value_counts()],).transpose().reset_index()

team_stat2= pd.merge(temp, temp2, on='winner', how='inner')
team_stat2= pd.merge(team_stat2, temp3, left_on='winner',right_on='index', how='inner')

team_stat2.rename(columns={'id_x':'batting_first','id_y':'batting_second'}, inplace= True)
team_stat2.drop(columns=['index'], inplace= True)

# finding win percent inning wise
team_stat2['win_percent_first']= team_stat2['batting_first']/ team_stat2['team1']
team_stat2['win_percent_second']= team_stat2['batting_second']/ team_stat2['team2']

# Aggregating the win percent
team_stat2['win_percent']=(team_stat2.batting_first+ team_stat2.batting_second)/ (team_stat2.team1+ team_stat2.team2)
team_stat2['total_wins']= team_stat2.batting_first + team_stat2.batting_second

# Sorting the values
team_stat2.sort_values(by='total_wins', inplace=True, ascending=False)
team_stat2.head()
plt.figure(figsize=(8,6))
sns.set_style('ticks')

sns.barplot(team_stat2.winner, team_stat2.total_wins ,label='Total wins', color='tab:blue')
plt.xticks(rotation=90)
plt.xlabel('IPL Team')
plt.ylabel('Total wins')

axes2= plt.twinx()
axes2.plot(team_stat2.winner,team_stat2.win_percent_first ,'g-o',label='Batting First')
axes2.plot(team_stat2.winner,team_stat2.win_percent_second ,'y-o',label='Fielding First')
axes2.plot(team_stat2.winner,team_stat2.win_percent ,'r-o',label='Average winning rate')

plt.ylabel('Winning Percent')
plt.yticks(np.arange(0,1.1,0.1) )
plt.title('Cliffhanger wins for each team')
plt.legend(loc=9 )
plt.show()
team_stat2
# Distribution for team batting first
temp= match.query('win_by_runs!=0')

plt.figure(figsize=(8,6))
plt.hist(temp.win_by_runs, bins=15)
plt.xticks(np.arange(0,160,10))
plt.xlabel('Margin of win by runs')
plt.ylabel('Frequency')
plt.title('Distribution for victory margin(in runs)')
plt.show()
match.query('batting_first_won==False').shape[0]
160/421
# Distribution for team batting second
temp = match.query('win_by_wickets>0').overs_2.apply(lambda x: int(x))

plt.figure(figsize=(8,6))
plt.hist(temp, bins= 16  )
plt.xticks(np.arange(5,21,1))
plt.xlabel('Overs')
plt.ylabel('Frequency')
plt.title('Which over did the chasing team win the match?')
plt.show()
# Most runs scored by a batsman
temp=batsman.sort_values(by='total_runs', ascending= False).head(7)
sns.set_style('dark')

plt.figure(figsize= (8,6))
sns.barplot(x= temp.batsman, y= temp.total_runs, color= 'tab:blue')
plt.ylabel('Total Runs scored')
plt.xlabel('Batsmen')
plt.title('Most runs scored by a batsman')
plt.yticks(np.arange(0,5600,500))

axes2= plt.twinx()
axes2.plot(temp.batsman, temp.boundary_percent*100, 'r-o', label='Percent of runs scored in boundary')
axes2.plot(temp.batsman, temp.strike_rate, 'y-o', label='Strike Rate')
plt.ylabel('Percent/ Strike Rate')
plt.yticks(np.arange(50,180,10))
plt.legend()
plt.show()
# Most runs scored by an opening batsman
temp=batsman.sort_values(by='opening_runs', ascending= False).head(7)
sns.set_style('dark', )

plt.figure(figsize= (8,6))
sns.barplot(x= temp.batsman, y= temp.opening_runs, color= 'tab:blue')
plt.ylabel('Total Runs scored')
plt.xlabel('Batsmen')
plt.title('Most runs scored by an opener')
plt.yticks(np.arange(0,5100,500))

axes2= plt.twinx()
axes2.plot(temp.batsman, temp.boundary_percent*100, 'r-o', label='Percent of runs scored in boundary')
axes2.plot(temp.batsman, temp.strike_rate, 'y-o', label='Strike Rate')
plt.ylabel('Percent/ Strike Rate')
plt.yticks(np.arange(50,180,10))

plt.legend()
plt.show()
# Most wickets taken
temp= bowler.sort_values(by='wickets_taken', ascending= False).head(7)

plt.figure(figsize=(9,6))
sns.barplot(x= temp.bowler, y= temp.wickets_taken, color= 'tab:blue')
plt.ylabel('Wickets taken')
plt.yticks(np.arange(0,190,20))

axes2= plt.twinx()
axes2.plot(temp.bowler, temp.bowl_lbw_percent, 'g-o', label='Percent of bowled/LBW dismissals')
axes2.plot(temp.bowler, temp.dot_ball_percent, 'r-o', label='Percent dot balls')
plt.ylabel('Percent', )
plt.yticks(np.arange(0,0.6,0.1))
plt.legend()
plt.title('Most wickets taken and their economy')

plt.show()
temp
# Bowlers with best economy
temp =bowler.query('balls_bowled > 120').sort_values(by='economy', ascending= True).head(10)
sns.set_style('darkgrid')
plt.figure(figsize=(12,6))
sns.lineplot(temp.bowler, temp.economy, color='tab:red', marker='o', label='Economy', sort= False)
sns.lineplot(temp.bowler, temp.death_econ, color='tab:green', marker='o', label='Economy in death overs (>18)', sort= False)
sns.lineplot(temp.bowler, temp.dot_ball_percent*6, color='tab:blue', marker='o', label='Average dot balls per over', sort= False)
plt.yticks(np.arange(0,11,1))
plt.ylabel('Economy/ Dot balls per over')
plt.xlabel('Bowler')
plt.title('Most economical bowlers')
plt.legend()
plt.show()
bowler.query('balls_bowled > 120').sort_values(by='death_balls', ascending= False)
# Win percentage for team batting first on each venue
major_city= ['Bangalore','Chandigarh','Chennai','Delhi','Hyderabad','Jaipur','Kolkata','Mumbai','Pune']

temp = match[match['city'].isin(major_city) ]
# Win percentage for team batting first on each venue
temp2 = temp.groupby(by='city').batting_first_won.mean()

plt.figure(figsize=(9,6))
sns.barplot(x=temp2.index, y=temp2, color='tab:blue')
plt.axhline(match.batting_first_won.mean(), color='r', label='Average winning rate for team batting first', linestyle='--')
plt.xlabel('Venue for the match')
plt.ylabel('Percent win for team batting first')
plt.title('Winning percent for team batting first across the venues')
plt.yticks(np.arange(0,0.7,0.05))
plt.legend()
plt.show()
major_teams= ['Hyderabad (Sunriser/Chargers)', 'Kolkata Knight Riders', 'Kings XI Punjab',
       'Royal Challengers Bangalore', 'Mumbai Indians',
       'Delhi (Capitals/ Daredevils)','Chennai Super Kings', 'Rajasthan Royals']
temp.pivot_table(index='city', columns='winner',values='batting_first_won', aggfunc='mean' )[major_teams]
plt.figure(figsize=(8,6))

sns.boxplot(x=temp.city, y=temp.runs_1,color= 'tab:blue' )
plt.axhline(match.runs_1.median(), color='r', linestyle='--')
plt.yticks(np.arange(0,325,20))
plt.ylabel('Runs scored on the venue')
plt.xlabel('Venue')
plt.title('Runs scored on each venue')
plt.show()
# Venue wise distribution for runs scored by team batting first
plt.figure(figsize=(9,6))
temp= match[match['city'].isin(major_city)]
sns.scatterplot(x= temp.runs_1, y= temp.city, hue= temp.batting_first_won, y_jitter= 1 )
plt.xticks(np.arange(60,270,20))
plt.legend(loc=(1.05,0.5))
plt.xlabel('Runs scored by team batting first')
plt.ylabel('Venue of the match')
plt.title('Runs scored by team batting first in each venue')
plt.show()
# Venue wise distribution for runs scored by team batting first
plt.figure(figsize=(9,6))
#temp= match[match['team1'].isin(major_teams)]
sns.scatterplot(x= match.runs_1, y= match.team1, hue= match.batting_first_won, y_jitter= 1 )
plt.xticks(np.arange(60,270,20))
plt.legend(loc=(1.03,0.8))
plt.xlabel('Runs scored by team batting first')
plt.ylabel('IPL Teams')
plt.title('Runs scored by team batting first vs winning record')
plt.show()
# Merging the match winner to the Ball dataframe
ball= pd.merge(ball, match.loc[:,['id','winner']], left_on='match_id', right_on= 'id', how='left')
temp = ball[ball.batting_team== ball.winner].groupby(by=['batsman', 'match_id']).sum()['batsman_runs'].reset_index()
s1= temp.groupby(by='batsman').sum()['batsman_runs'].sort_values(ascending= False).head(15)

temp1= ball.groupby(by=['batting_team','match_id', 'batsman']).sum()['batsman_runs'].reset_index()
temp2= pd.merge(temp1, match[['id','winner']], left_on= 'match_id', right_on='id', how='left')
temp2['is_win'] = temp2.winner== temp2.batting_team
temp3= temp2.query('batsman_runs>50').groupby(by='batsman').mean()['is_win'].sort_values(ascending= False).reset_index()
temp4= pd.merge(temp3[temp3.batsman.isin(s1.index)], s1.reset_index(),on='batsman', how='inner').sort_values(by='batsman_runs', ascending= False)
sns.set_style('dark')
plt.figure(figsize=(8,6))
sns.barplot(y= 'batsman_runs', x= 'batsman', data= temp4, color= 'tab:blue')
plt.xticks(rotation=90)
plt.title('Most runs scored in winning cause and win percent')
#plt.yticks('Runs scored by batsmen in winning cause')

axes= plt.twinx()
axes.plot(temp4.batsman, temp4.is_win,  'r--o', label='Win percent when batsman scored >50 runs')
plt.ylabel('Win percent when batsman scored more than 50 runs')
plt.yticks(np.arange(0.4,0.90,0.05))

plt.legend()
plt.show()
s= match.player_of_match.value_counts().head(10)
plt.figure(figsize=(8,6))
sns.barplot(y= s.index, x= s, color= 'tab:blue')
plt.xlabel('Number of awards')
plt.ylabel('Player name')
plt.title('Most number of Player of the Match awards')
plt.xticks(np.arange(0,24,2))
plt.show()
temp= ball.groupby(by=['batsman', 'match_id']).sum()['batsman_runs'].reset_index()

temp['>25']= temp.batsman_runs> 25
s1= temp.groupby(by='batsman').mean()['>25'].sort_values(ascending= False)
s2= temp.groupby(by='batsman').count()['match_id'].sort_values(ascending= False)

temp['>50']= temp.batsman_runs> 50
s3= temp.groupby(by='batsman').mean()['>50'].sort_values(ascending= False)

df1= pd.DataFrame([s1,s3,s2]).transpose()
temp2= df1.query('match_id>=10').head(10)

plt.figure(figsize=(8,6))
sns.barplot(x= temp2.index, y= temp2.match_id, color= 'tab:blue')
plt.xticks(rotation=90)
plt.ylabel('Total matches played')
plt.title('Batsman consistency across matches')

axes= plt.twinx()
axes.plot(temp2.index, temp2['>25'], 'g--o', label='Percent when batsman scored > 25')
axes.plot(temp2.index, temp2['>50'], 'r--o', label='Percent when batsman scored > 50')
plt.legend()
plt.xticks(rotation=90)
plt.xlabel('Player name')
plt.ylabel('Percent')
plt.show()
temp2