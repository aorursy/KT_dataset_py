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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go
file=pd.read_csv(r'/kaggle/input/ipldata/deliveries.csv',encoding='latin1')

file.head()
file2=pd.read_csv('/kaggle/input/ipldata/matches.csv')

file2.head()
file2.team1.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)

file2.team2.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)

file2.winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)

file2.venue.replace({'Feroz Shah Kotla Ground':'Feroz Shah Kotla',

                    'M Chinnaswamy Stadium':'M. Chinnaswamy Stadium',

                    'MA Chidambaram Stadium, Chepauk':'M.A. Chidambaram Stadium',

                     'M. A. Chidambaram Stadium':'M.A. Chidambaram Stadium',

                     'Punjab Cricket Association IS Bindra Stadium, Mohali':'Punjab Cricket Association Stadium',

                     'Punjab Cricket Association Stadium, Mohali':'Punjab Cricket Association Stadium',

                     'IS Bindra Stadium':'Punjab Cricket Association Stadium',

                    'Rajiv Gandhi International Stadium, Uppal':'Rajiv Gandhi International Stadium',

                    'Rajiv Gandhi Intl. Cricket Stadium':'Rajiv Gandhi International Stadium'},regex=True,inplace=True)
city_counts=file2.groupby('city').apply(lambda x:x['city'].count()).reset_index(name='Match Counts')

top_cities_order=city_counts.sort_values(by='Match Counts',ascending=False)

top_cities=top_cities_order[:20]

print('Top 15 Cities with the maximum number of Matches Played:\n',top_cities)

plt.figure(figsize=(9,9))

plt.pie(top_cities['Match Counts'],labels=top_cities['city'],autopct='%1.1f%%', startangle=30)

plt.axis('equal')

plt.title('Top Cities that have hosted IPL Matches',size=20)
venue_counts=file2.groupby('venue').apply(lambda x:x['venue'].count()).reset_index(name='Match Counts')

top_venues_order=venue_counts.sort_values(by='Match Counts',ascending=False)

top_venues=top_venues_order[:20]

print('Top 20 Stadiums with the maximum number of Matches Played:\n',top_venues)

plt.figure(figsize=(9,9))

plt.pie(top_venues['Match Counts'],labels=top_venues['venue'],autopct='%1.1f%%', startangle=40)

plt.axis('equal')

plt.title('Top Stadiums that have hosted IPL Matches',size=20)
batting_tot=file.groupby('batsman').apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Runs')

batting_sorted=batting_tot.sort_values(by='Runs',ascending=False)

top_batsmen=batting_sorted[:10] 

print('The Top 10 Batsmen in thr Tournament are:\n',top_batsmen)

fig = px.bar(top_batsmen, x='batsman', y='Runs',

             hover_data=['batsman'], color='Runs',title='Top 10 Batsmen in IPL- Seasons 2008-2019')

fig.show()
batting_ings=file.groupby(['match_id','batsman']).apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Innings Runs')

batting_ings_sorted=batting_ings.sort_values(by='Innings Runs',ascending=False)

top_batsmen_scores=batting_ings_sorted[:10] 

batsman_ball_faced=file.groupby(['match_id','batsman']).apply(lambda x:x['batsman_runs'].count()).reset_index(name='Balls Faced')

batsmen_performance=pd.merge(top_batsmen_scores,batsman_ball_faced,how='inner',left_on=['match_id','batsman'],right_on=['match_id','batsman'])

batsmen_performance['Strike Rate for Match']=batsmen_performance['Innings Runs']*100/batsmen_performance['Balls Faced']

batsmen_innings=pd.merge(batsmen_performance,file,how='inner',left_on=['match_id','batsman'],right_on=['match_id','batsman'])

batsmen_innings_req=batsmen_innings.iloc[:,1:8]

batsmen_innings_req_2=batsmen_innings_req.drop_duplicates()

print('The Top 10 Batting Performances in the IPL History are:\n',batsmen_innings_req_2)

x=batsmen_innings_req_2['batsman']

y1=batsmen_innings_req_2['Strike Rate for Match']

y2=batsmen_innings_req_2['Innings Runs']

plt.figure(figsize=(10,5))

plt.scatter(x,y1)

plt.scatter(x,y2)

plt.xlabel('Batsmen',size=15)

plt.ylabel('Strike Rate/Innings Score',size=15)

plt.title('IPL Best batting performances in a Match',size=20)

plt.xticks(rotation=60)

plt.legend(['Strike Rate','Runs'],prop={'size':20})
#Run Out is not considered as a wicket in the Bowler's account- hence we shall be removing them first

bowling_wickets=file[file['dismissal_kind']!='run out']

bowling_tot=bowling_wickets.groupby('bowler').apply(lambda x:x['dismissal_kind'].dropna()).reset_index(name='Wickets')

bowling_wick_count=bowling_tot.groupby('bowler').count().reset_index()

bowling_top=bowling_wick_count.sort_values(by='Wickets',ascending=False)

top_bowlers=bowling_top.loc[:,['bowler','Wickets']][0:10] 

print('The Top Wicket Takers in the Tournament are:\n',top_bowlers)

fig = px.bar(top_bowlers, x='bowler', y='Wickets',

             hover_data=['bowler'], color='Wickets',title='Top 10 Bowlers in IPL- Seasons 2008-2019')

fig.show()
#Run Out is not considered as a wicket in the Bowler's account- hence we shall be removing them first

match_bowling_tot=bowling_wickets.groupby(['match_id','bowler']).apply(lambda x:x['dismissal_kind'].dropna()).reset_index(name='Wickets')

match_bowling_wick_count=match_bowling_tot.groupby(['match_id','bowler']).count().reset_index()

match_bowling_top=match_bowling_wick_count.sort_values(by='Wickets',ascending=False)

match_top_bowlers=match_bowling_top.loc[:,['match_id','bowler','Wickets']][0:10] 

match_bowling_runs=file.groupby(['match_id','bowler']).apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Runs Conceeded')

match_bowler_performance=pd.merge(match_top_bowlers,match_bowling_runs,how='inner',left_on=['match_id','bowler'],right_on=['match_id','bowler'])

match_bowler_performance['Runs per Wicket']=match_bowler_performance['Runs Conceeded']/match_bowler_performance['Wickets']

bowler_innings=pd.merge(match_bowler_performance,file,how='inner',left_on=['match_id','bowler'],right_on=['match_id','bowler'])

bowler_innings_req=bowler_innings.iloc[:,1:8]

bowler_innings_req_2=bowler_innings_req.drop_duplicates()

print('The Top 10 Batting Performances in the IPL History are:\n',bowler_innings_req_2)

x=bowler_innings_req_2['bowler']

y1=bowler_innings_req_2['Wickets']

y2=bowler_innings_req_2['Runs per Wicket']

plt.figure(figsize=(10,5))

plt.scatter(x,y1)

plt.plot(x,y2,'r')

plt.xlabel('Bowlers',size=15)

plt.ylabel('Runs per Wicket/Wickets',size=15)

plt.title('IPL Best bowling performances in a Match',size=20)

plt.xticks(rotation=60)

plt.legend(['Runs per Wicket','Wickets'],prop={'size':15})
#Creating a list of the best fielders- Considering Catch,Run Out and Stumpings

fielder_list=file.groupby('fielder').apply(lambda x:x).dropna().reset_index()

fielder_list_count=fielder_list.groupby('fielder').count()

fielder_list_counts=fielder_list_count['dismissal_kind'].reset_index(name='Dismissals')

fielder_list_max=fielder_list_counts.sort_values(by='Dismissals',ascending=False)

top_fielders=fielder_list_max[0:10]

print('The Best Fielders(and WicketKeepers) in the Torunament are:\n',top_fielders)



fig = px.bar(top_fielders, x='fielder', y='Dismissals',

             hover_data=['fielder'], color='Dismissals',title='Top 10 Fielders in IPL- Seasons 2008-2019')

fig.show()
Target_run=1000

batting_tot=file.groupby('batsman').apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Runs')

batsman_balls_faced=file.groupby('batsman').count()

batsman_balls_faced_count=batsman_balls_faced['ball'].reset_index(name='Balls Faced')

batsman_runs_balls=pd.merge(batting_tot,batsman_balls_faced_count,left_on='batsman',right_on='batsman',how='outer')

batsman_strike_rate=batsman_runs_balls.groupby(['batsman','Runs']).apply(lambda x:((x['Runs'])/(x['Balls Faced']))*100).reset_index(name='Strike Rate')

plt.scatter(batsman_strike_rate['Runs'],batsman_strike_rate['Strike Rate'])

plt.plot(np.mean(batsman_strike_rate['Strike Rate']),'r')

plt.xlabel('Batsman Runs',size=15)

plt.ylabel('Strike Rate',size=15)

plt.title('Overall Runs vs Strike Rate Analysis',size=25)

plt.show()

batsman_strike_rate_list=batsman_strike_rate.sort_values(by='Strike Rate',ascending=False)

batsman_strike_rate_above_target_runs=batsman_strike_rate_list[batsman_strike_rate_list['Runs']>=Target_run]

top_strike_rate_batsman=batsman_strike_rate_above_target_runs.loc[:,['batsman','Runs','Strike Rate']][0:10]

print('The Top 10 batsmen having highest strike rate, scoring atleast {} Runs:\n'.format(Target_run),top_strike_rate_batsman)

plt.plot(top_strike_rate_batsman['batsman'],top_strike_rate_batsman['Strike Rate'],color='r')

plt.scatter(top_strike_rate_batsman['batsman'],top_strike_rate_batsman['Strike Rate'],color='g')

plt.xlabel('Batsman',size=15)

plt.ylabel('Strike Rate',size=15)

plt.title('Top 10 Batsmen Strike Rate Analysis',size=25)

plt.xticks(rotation=60)
Ball_Limit=1000

bowling_runs=file.groupby('bowler').apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Runs Conceeded')

bowling_balls=file.groupby('bowler').count()

bowled_balls=bowling_balls['ball'].reset_index(name='Balls Bowled')

bowler_stats=pd.merge(bowling_runs,bowled_balls,left_on='bowler',right_on='bowler',how='outer')

bowler_economy_rate=bowler_stats.groupby(['bowler','Balls Bowled']).apply(lambda x:(((x['Runs Conceeded'])/(x['Balls Bowled']))*6)).reset_index(name='Economy Rate')

plt.scatter(bowler_economy_rate['Balls Bowled'],bowler_economy_rate['Economy Rate'],color='g')

plt.xlabel('Balls Bowled',size=15)

plt.ylabel('Economy Rate',size=15)

plt.title('Balls vs Economy Rate Analysis',size=25)

plt.show()

bowler_best_economy_rate=bowler_economy_rate.sort_values(by='Economy Rate',ascending=True)

bowler_best_economy_rate_condition=bowler_best_economy_rate[bowler_best_economy_rate['Balls Bowled']>=Ball_Limit]

top_10_economy=bowler_best_economy_rate_condition.loc[:,['bowler','Balls Bowled','Economy Rate']][0:10]

print('The Top 10 bowlers having best economy rate, bowling atleast {} balls:\n'.format(Ball_Limit),top_10_economy)

plt.plot(top_10_economy['bowler'],top_10_economy['Economy Rate'],color='y')

plt.scatter(top_10_economy['bowler'],top_10_economy['Economy Rate'],color='b')

plt.xlabel('Bowlers',size=15)

plt.ylabel('Economy Rate',size=15)

plt.title('Top 10 Bowler Economy Rate Analysis',size=25)

plt.xticks(rotation=60)
motm=file2.groupby('player_of_match').apply(lambda x:x['player_of_match'].count()).reset_index(name='Man of the Match Awards')

motm_sort=motm.sort_values(by='Man of the Match Awards',ascending=False)

motm_top=motm_sort[0:15]

plt.plot(motm_top['player_of_match'],motm_top['Man of the Match Awards'],color='b')

plt.bar(motm_top['player_of_match'],motm_top['Man of the Match Awards'],color='y')

plt.xlabel('Players')

plt.ylabel('Man of the Match Award Count')

plt.title('Top 15 Players who have won most the Man of the Match trophies',size=15)

plt.xticks(rotation=60)

batting_factor=0.5

bowling_factor=15.0

fielding_factor=10.0

all_rounding_1=pd.merge(batting_sorted,bowling_top,left_on='batsman',right_on='bowler',how='inner')

all_rounding_2=pd.merge(all_rounding_1,fielder_list_max,left_on='batsman',right_on='fielder',how='left')

all_rounding_performance=all_rounding_2.groupby(['batsman','Runs','Wickets','Dismissals']).apply(lambda x:(((x['Runs'])*batting_factor)+((x['Wickets'])*bowling_factor)+((x['Dismissals'])*fielding_factor))).reset_index(name='Overall Score')

best_all_round_performance=all_rounding_performance.sort_values(by='Overall Score',ascending=False)

best_overall=best_all_round_performance.loc[:,['batsman','Runs','Wickets','Dismissals','Overall Score']][0:10]

print('The top 10 best players overall are:\n',best_overall)

plt.figure(figsize=(10,10))

plt.plot(best_overall['batsman'],best_overall['Runs']*batting_factor,'g')

plt.plot(best_overall['batsman'],best_overall['Wickets']*bowling_factor,'r')

plt.plot(best_overall['batsman'],best_overall['Dismissals']*fielding_factor,'y')

plt.plot(best_overall['batsman'],best_overall['Overall Score'])

plt.xlabel('The Top 10 performers',size=15)

plt.ylabel('Scoring Units',size=15)

plt.xticks(rotation=60)

plt.title('Overall Performance by Top 10 Performers in IPL-2008-2019',size=20)

plt.legend(['Run Points','Wicket Points','Dismissal Points','Overall Score'])
batsman_list_req=['MS Dhoni','V Kohli','SC Ganguly']

batsman=file[file.batsman.isin(batsman_list_req)]

batsman_run=batsman.groupby(['match_id','batsman']).apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Runs')

#bat_list=batsman.batsman.unique()

plt.figure(figsize=(30,10))

for name in batsman_list_req:

    batsman_check=batsman_run[batsman_run.batsman==name]

    batsman_check.index = np.arange(1, len(batsman_check) + 1)

    x=batsman_check.index

    y=batsman_check.Runs

    plt.bar(x,y)

plt.legend(batsman_list_req,prop={'size':20})

plt.title("Innings Total across all Seasons- 2008-2019",fontsize=60)

plt.xlabel("Total Matches Played",fontsize=30)

plt.ylabel("Runs Scored in a Match",fontsize=30)

plt.show()
batsman_list_req=['V Kohli']

opposition_team='Kolkata Knight Riders'

ball_limit=12

cond_1_1=file.batsman.isin(batsman_list_req)

cond_1_2=file.bowling_team==opposition_team

batsman_team=file[(cond_1_1) & (cond_1_2)]

batsman_team_run=batsman_team.groupby(['match_id','batsman','bowling_team']).apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Runs')

bowling_runs=batsman_team.groupby('bowler').apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Runs Conceeded')

bowling_balls=batsman_team.groupby('bowler').count()

bowled_balls=bowling_balls['ball'].reset_index(name='Balls Bowled')

bowled_balls_limit=bowled_balls[bowled_balls['Balls Bowled']>=ball_limit]

bowler_stats=pd.merge(bowling_runs,bowled_balls_limit,left_on='bowler',right_on='bowler',how='inner')

bowler_economy_rate=bowler_stats.groupby(['bowler','Balls Bowled']).apply(lambda x:(((x['Runs Conceeded'])/(x['Balls Bowled']))*6)).reset_index(name='Economy Rate')

bowler_best_to_worst_1=bowler_economy_rate.sort_values(by='Economy Rate',ascending=True)

bowler_best_to_worst=bowler_best_to_worst_1.loc[:,['bowler','Balls Bowled','Economy Rate']]

plt.figure(figsize=(30,10))

batsman_team_run.index = np.arange(1, len(batsman_team_run) + 1)

x=batsman_team_run.index

y=batsman_team_run.Runs

plt.bar(x,y)

plt.plot(x,y,'r')

plt.title("{} innings wise score against {} across- 2008-2019".format(batsman_list_req[0],opposition_team),fontsize=30)

plt.xlabel("Total Matches Played",fontsize=30)

plt.ylabel("Runs Scored in a Match",fontsize=30)

plt.legend(['Runs'],prop={'size':30})

plt.show()

print('The runs scored in matches: \n',y)

print('---------------------------------------------------------------------------------------------')

print('The Economy rate of the various bowlers of {} against the {} (best to worst)\n'.format(opposition_team,batsman_list_req[0]),bowler_best_to_worst)
first_innins_run=file[file['inning']==1]

team_innings_run=first_innins_run.groupby(['batting_team','match_id']).apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Innings Total')

team_innings_avg=team_innings_run.groupby('batting_team').apply(lambda x:np.mean(x['Innings Total'])).reset_index(name='Innings Average')

plt.plot(team_innings_avg['batting_team'],team_innings_avg['Innings Average'],'b')

second_innins_run=file[file['inning']==2]

team_innings_run=second_innins_run.groupby(['batting_team','match_id']).apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Innings Total')

team_innings_avg=team_innings_run.groupby('batting_team').apply(lambda x:np.mean(x['Innings Total'])).reset_index(name='Innings Average')

plt.plot(team_innings_avg['batting_team'],team_innings_avg['Innings Average'],'r')

plt.xticks(rotation=90)

plt.xlabel('IPL Teams',size=15)

plt.ylabel('Innings Average',size=15)

plt.title('Team wise Batting Average in IPL- Seasons 2008-2019',size=20)
first_innins_score=file[file['inning']==1]

team_innings_score=first_innins_score.groupby(['bowling_team','match_id']).apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Innings Total')

team_innings_score_avg=team_innings_score.groupby('bowling_team').apply(lambda x:np.mean(x['Innings Total'])).reset_index(name='Innings Average')

plt.plot(team_innings_score_avg['bowling_team'],team_innings_score_avg['Innings Average'],'b')

second_innins_score=file[file['inning']==2]

team_innings_second_score=second_innins_score.groupby(['bowling_team','match_id']).apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Innings Total')

team_second_innings_score_avg=team_innings_second_score.groupby('bowling_team').apply(lambda x:np.mean(x['Innings Total'])).reset_index(name='Innings Average')

plt.plot(team_second_innings_score_avg['bowling_team'],team_second_innings_score_avg['Innings Average'],'r')

plt.xticks(rotation=90)

plt.legend(['First Innings','Second Innings'],prop={'size':10})

plt.xlabel('IPL Teams',size=15)

plt.ylabel('Innings Average',size=15)

plt.title('Team wise Bowling Average in IPL- Seasons 2008-2019',size=20)
win_runs=file2.groupby('winner').apply(lambda x:np.average(x['win_by_runs'])).reset_index(name='Win By Runs Average')

win_wickets=file2.groupby('winner').apply(lambda x:np.average(x['win_by_wickets'])).reset_index(name='Win By Wickets Average')

plt.figure(figsize=(7,7))

plt.plot(win_runs['winner'],win_runs['Win By Runs Average'],color='b')

plt.plot(win_wickets['winner'],win_wickets['Win By Wickets Average'],color='r')

plt.xlabel('Teams',size=15)

plt.xticks(rotation=90)

plt.ylabel('Winning Metrics',size=15)

plt.legend(['Win by Runs','Win by Wickets'])

plt.title('Teams Average winning by Runs/Wickets Summary')
Current_teams=['Chennai Super Kings','Mumbai Indians','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad','Kolkata Knight Riders','Royal Challengers Bangalore','Kings XI Punjab']

team_1_filter=file2[file2.team1.isin(Current_teams)]

team_2_filter=team_1_filter[team_1_filter.team2.isin(Current_teams)]

teams_filter=team_2_filter[team_2_filter.winner.isin(Current_teams)]

head_to_head_matches=teams_filter.groupby(['team1','team2','winner']).apply(lambda x:x['winner'].count()).reset_index(name='Winning Counts')

head_to_head_matches['Game']=head_to_head_matches['team1']+' vs. '+head_to_head_matches['team2']

head_to_head_matches.loc[:,['Game','winner','Winning Counts']]

heatmap1_data = pd.pivot_table(head_to_head_matches, values='Winning Counts', 

                     index=['Game'], 

                     columns='winner')

fig = plt.figure()

fig, ax = plt.subplots(1,1, figsize=(5,15))

g=sns.heatmap(heatmap1_data,annot=True, cmap="YlGnBu",fmt='g')

ax.set_title('The Head-to-Head Performace Matrix of Teams in IPL',size=20)

ax.set_xlabel('IPL Teams',size=15)

ax.set_ylabel('Match',size=15)
venue_win=file2.groupby(['venue','winner']).apply(lambda x:x['winner'].count()).reset_index(name='Match Wins')

venue_win_pvt=pd.pivot(venue_win,values='Match Wins',index='venue',columns='winner')

venue_win_pvt.replace(np.NaN,0)

plt.figure(figsize=(20,10))

htmp=sns.heatmap(venue_win_pvt,annot=True,fmt='g',cmap='PuBuGn')

plt.xlabel('Teams',size=25)

plt.ylabel('Venues',size=25)

plt.title('Team wise wins at the Venues',size=45)
venue_mom=file2.groupby(['venue','player_of_match']).apply(lambda x:x['player_of_match'].count()).reset_index(name='MoM_Winner')

venue_mom_sort=venue_mom.sort_values(by=['venue','MoM_Winner'],ascending=[True,False])

venue_mom_count_max=venue_mom_sort.groupby(['venue']).apply(lambda x:np.max(x['MoM_Winner'])).reset_index(name='MoM_Winner')

venue_best=pd.merge(venue_mom,venue_mom_count_max,how='inner',left_on=['venue','MoM_Winner'],right_on=['venue','MoM_Winner'])

venue_best_multiple_pivot=pd.pivot(venue_best,values='MoM_Winner',index='player_of_match',columns='venue')

plt.figure(figsize=(10,25))

sns.heatmap(venue_best_multiple_pivot,annot=True,fmt='g',cmap='Wistia')

plt.xlabel('IPL Venues',size=25)

plt.ylabel('Players',size=25)

plt.title('Players with the Best Performance at Venues',size=25)
venue_toss=teams_filter.groupby(['venue','toss_decision']).apply(lambda x:x['toss_decision'].count()).reset_index(name='Toss Decision Counts')

heatmap2_data = pd.pivot_table(venue_toss, values='Toss Decision Counts', 

                     index=['venue'], 

                     columns='toss_decision')

fig = plt.figure()

fig, ax = plt.subplots(1,1, figsize=(5,15))

g=sns.heatmap(heatmap2_data,annot=True, cmap="Blues",fmt='g')

g.xaxis.set_ticks_position("top")

ax.set_title('The Toss Decisions taken by Venue Heatmap-in IPL',size=20)
venue_toss_result=teams_filter.groupby(['venue','toss_decision']).apply(lambda x:np.sum(np.where(x['toss_winner']==x['winner'],1,0))).reset_index(name='Toss Winner Wins Match')

merged_venue_data=pd.merge(venue_toss_result,venue_toss,how='inner',left_on=['venue','toss_decision'],right_on=['venue','toss_decision'])

merged_venue_data['Toss Winner Lose Match']=merged_venue_data['Toss Decision Counts']-merged_venue_data['Toss Winner Wins Match']

merged_data_arranged=merged_venue_data.loc[:,['venue','toss_decision','Toss Decision Counts','Toss Winner Wins Match','Toss Winner Lose Match']]

merged_data_arranged
heatmap3_data = pd.pivot_table(merged_venue_data, values=['Toss Winner Wins Match','Toss Winner Lose Match'], 

                     index=['venue'], 

                     columns='toss_decision')

fig=plt.figure()

fig,ax1=plt.subplots(1,1,figsize=(8,20))

g=sns.heatmap(heatmap3_data,annot=True,cmap='YlOrBr',fmt='g')

g.xaxis.set_ticks_position("top")

g.set_xticklabels(['Won Toss, Bat First- Match Lost','Won Toss, Field First- Match Lost','Won Toss, Bat First- Match Won','Won Toss, Field First- Match Won'],rotation=90)

g.set_xlabel('Toss Winner- Match Win vs Loss Heatmap',size=15)

g.set_ylabel('Stadium',size=20)