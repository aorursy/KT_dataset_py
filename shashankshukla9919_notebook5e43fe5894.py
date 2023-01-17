import numpy as np
import pandas as pd
matches = pd.read_csv("D:/Widhya WPL/10th Oct/deliveries.csv")
matches.head()
deliveries = pd.read_csv("D:/Widhya WPL/10th Oct/matches.csv")
deliveries.head()
deliveries.isnull().sum()
deliveries.columns
matches.isnull().sum()
matches.columns
deliveries.drop(['umpire3'], axis = 1, inplace = True)
print(deliveries['winner'].unique())
print(deliveries['city'].unique())
deliveries.team1.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
deliveries.team2.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
deliveries.winner.replace({'Rising Pune Supergiants' : 'Rising Pune Supergiant'},regex=True,inplace=True)
deliveries.venue.replace({'Feroz Shah Kotla Ground':'Feroz Shah Kotla',
                    'M Chinnaswamy Stadium':'M. Chinnaswamy Stadium',
                    'MA Chidambaram Stadium, Chepauk':'M.A. Chidambaram Stadium',
                     'M. A. Chidambaram Stadium':'M.A. Chidambaram Stadium',
                     'Punjab Cricket Association IS Bindra Stadium, Mohali':'Punjab Cricket Association Stadium',
                     'Punjab Cricket Association Stadium, Mohali':'Punjab Cricket Association Stadium',
                     'IS Bindra Stadium':'Punjab Cricket Association Stadium',
                    'Rajiv Gandhi International Stadium, Uppal':'Rajiv Gandhi International Stadium',
                    'Rajiv Gandhi Intl. Cricket Stadium':'Rajiv Gandhi International Stadium'},regex=True,inplace=True)
deliveries.replace('Bangalore','Bengaluru', inplace = True)
#fill missing values
deliveries['city'].fillna(deliveries['venue'], inplace = True)
deliveries['winner'].fillna(deliveries['result'], inplace = True)
deliveries['player_of_match'].fillna(deliveries['result'], inplace = True)
deliveries['umpire1'].fillna('unknown', inplace = True)
deliveries['umpire2'].fillna('unknown', inplace = True)
def annotation_plot(ax,w,h):                                    # function to add data to plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for p in ax.patches:
        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))
total_matches = deliveries.groupby('season')['id'].count()
plt.figure(figsize=(10,5))
ax = sns.countplot("season", data = deliveries, palette='viridis')
plt.title('Total number of matches till 2020 (2008-2019)')
plt.ylabel('Number of Matches.')
annotation_plot(ax,0.08,1)
plt.show()
plt.figure(figsize=(12,7))
ax = sns.countplot("winner", data = deliveries, order = deliveries['winner'].value_counts().index,palette='viridis')
plt.title("Total number of wins by each team till 2020")
plt.xticks(rotation=45, ha = 'right')
plt.ylabel('Number of matches')
annotation_plot(ax,0.08,1)
plt.show()
max_times_winner = deliveries.groupby('season')['winner'].value_counts()
max_times_winner
groups = max_times_winner.groupby('season')
fig = plt.figure()
count = 1

for year, group in groups:
    ax = fig.add_subplot(4,3,count)
    ax.set_title(year)
    ax = group[year].plot.bar(figsize = (10,15), width = 0.8)
    
    count+=1;
    
    plt.xlabel('')
    plt.yticks([])
    plt.ylabel('Matches Won')
    
    total_of_matches = []
    for i in ax.patches:
        total_of_matches.append(i.get_height())
    total = sum(total_of_matches)
    
    for i in ax.patches:
        ax.text(i.get_x()+0.2, i.get_height()-1.5,s= i.get_height(),color="black",fontweight='bold')
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,7))
ax = sns.countplot("winner", data = deliveries, hue = 'toss_decision',order = deliveries['winner'].value_counts().index,palette='viridis')
plt.title("Total number of wins for every team till 2020")
plt.xticks(rotation=45, ha = 'right')
plt.ylabel('Number of Matches')
annotation_plot(ax,0.08,1)
plt.show()
Total_matches_played = deliveries['team1'].value_counts() + deliveries['team2'].value_counts()

toss_won = deliveries['toss_winner'].value_counts()
toss_win_success_rate = (toss_won/Total_matches_played)*100
toss_win_success_rate_sort = toss_win_success_rate.sort_values(ascending = False)
plt.figure(figsize = (10,5))
ax = sns.barplot(x =toss_win_success_rate_sort.index, y = toss_win_success_rate_sort,palette='viridis')
plt.xticks(rotation = 45, ha = 'right')
plt.ylabel('Toss Win success ratio.')
annotation_plot(ax,0.08,1)
plt.show()
plt.figure(figsize=(12,6))

ax = sns.countplot("player_of_match", data = deliveries,order = deliveries['player_of_match'].value_counts()[:20].index,palette='viridis')
plt.title("Total number of Player of the match. ")
plt.xticks(rotation=90, ha = 'right')
plt.ylabel('Number of Player of the match')
plt.xlabel('Name of the top 20 Player of the match.')
annotation_plot(ax,0.08,1)
plt.show()
total_matches
matches_won = deliveries.groupby('winner').count()
total_matches = deliveries['team1'].value_counts()+ deliveries['team2'].value_counts()

matches_won['Total matches'] = total_matches
win_df = matches_won[["Total matches","result"]]
sucess_ratio = round((matches_won['id']/total_matches),4)*100
sucess_ratio_sort = sucess_ratio.sort_values(ascending = False)
plt.figure(figsize = (10,7))
ax = sns.barplot(x = sucess_ratio_sort.index, y = sucess_ratio_sort, palette='viridis' )
annotation_plot(ax,0.08,1)
plt.xticks(rotation=45, ha = 'right')
plt.ylabel('Success rate of wining')
plt.show()
each_season_winner = deliveries.groupby('season')['season','winner'].tail(1)
each_season_winner_sort = each_season_winner.sort_values('season',ascending = True)
sns.countplot('winner', data = each_season_winner_sort)
plt.xticks(rotation = 45, ha = 'right')
plt.ylabel('Number of seasons won by any team.')
plt.show()
batting_tot=matches.groupby('batsman').apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Runs')
batting_sorted=batting_tot.sort_values(by='Runs',ascending=False)
top_batsmen=batting_sorted[:10] 
print('The Top 10 Batsmen in thr Tournament are:\n',top_batsmen)
plt.bar(top_batsmen['batsman'],top_batsmen['Runs'])
plt.scatter(top_batsmen['batsman'],top_batsmen['Runs'],color='r')
plt.xticks(rotation=60)
plt.xlabel('Top 10 Batsmen',size=10)
plt.ylabel('Runs Scored',size=10)
plt.title('Top 10 Batsmen in IPL- Seasons till 2020',size=20)
batting_ings=matches.groupby(['match_id','batsman']).apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name='Innings Runs')
batting_ings_sorted=batting_ings.sort_values(by='Innings Runs',ascending=False)
top_batsmen_scores=batting_ings_sorted[:10] 
batsman_ball_faced=matches.groupby(['match_id','batsman']).apply(lambda x:x['batsman_runs'].count()).reset_index(name='Balls Faced')
batsmen_performance=pd.merge(top_batsmen_scores,batsman_ball_faced,how='inner',left_on=['match_id','batsman'],right_on=['match_id','batsman'])
batsmen_performance['Strike Rate for Match']=batsmen_performance['Innings Runs']*100/batsmen_performance['Balls Faced']
batsmen_innings=pd.merge(batsmen_performance,matches,how='inner',left_on=['match_id','batsman'],right_on=['match_id','batsman'])
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
bowling_wickets=matches[matches['dismissal_kind']!='run out']
bowling_tot=bowling_wickets.groupby('bowler').apply(lambda x:x['dismissal_kind'].dropna()).reset_index(name='Wickets')
bowling_wick_count=bowling_tot.groupby('bowler').count().reset_index()
bowling_top=bowling_wick_count.sort_values(by='Wickets',ascending=False)
top_bowlers=bowling_top.loc[:,['bowler','Wickets']][0:10] 
print('The Top Wicket Takers in the Tournament are:\n',top_bowlers)
plt.scatter(top_bowlers['bowler'],top_bowlers['Wickets'],color='r')
plt.plot(top_bowlers['bowler'],top_bowlers['Wickets'],color='g')
plt.xticks(rotation=60)
plt.xlabel('Top 10 Bowlers',size=15)
plt.ylabel('Wickets Taken',size=15)
plt.title('Top 10 Bowlers in IPL- Seasons 2008-2019',size=20)
#Run Out is not considered as a wicket in the Bowler's account- hence we shall be removing them first
match_bowling_tot=bowling_wickets.groupby(['match_id','bowler']).apply(lambda x:x['dismissal_kind'].dropna()).reset_index(name='Wickets')
match_bowling_wick_count=match_bowling_tot.groupby(['match_id','bowler']).count().reset_index()
match_bowling_top=match_bowling_wick_count.sort_values(by='Wickets',ascending=False)
match_top_bowlers=match_bowling_top.loc[:,['match_id','bowler','Wickets']][0:10] 
match_bowling_runs=matches.groupby(['match_id','bowler']).apply(lambda x:np.sum(x['total_runs'])).reset_index(name='Runs Conceeded')
match_bowler_performance=pd.merge(match_top_bowlers,match_bowling_runs,how='inner',left_on=['match_id','bowler'],right_on=['match_id','bowler'])
match_bowler_performance['Runs per Wicket']=match_bowler_performance['Runs Conceeded']/match_bowler_performance['Wickets']
bowler_innings=pd.merge(match_bowler_performance,matches,how='inner',left_on=['match_id','bowler'],right_on=['match_id','bowler'])
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
city_counts= deliveries.groupby('city').apply(lambda x:x['city'].count()).reset_index(name='Match Counts')
top_cities_order=city_counts.sort_values(by='Match Counts',ascending=False)
top_cities=top_cities_order[:10]
print('Top 10 Cities with the maximum number of Matches Played:\n',top_cities)
plt.figure(figsize=(7,7))
plt.pie(top_cities['Match Counts'],labels=top_cities['city'],autopct='%1.1f%%', startangle=30)
plt.axis('equal')
plt.title('Top Cities that have hosted IPL Matches',size=10)
plt.figure(figsize = (12,6))
venue = deliveries[['city','winner','season']]
venue_season = venue[venue['season'] == 2018]
ax = sns.countplot('city', data = venue_season, hue = 'winner' )
plt.xticks(rotation=30, ha = 'right')
plt.ylabel('Number of matches.')
plt.show()
Current_teams=['Chennai Super Kings','Mumbai Indians','Rajasthan Royals','Delhi Capitals','Sunrisers Hyderabad','Kolkata Knight Riders','Royal Challengers Bangalore','Kings XI Punjab']
team_1_filter=deliveries[deliveries.team1.isin(Current_teams)]
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
venue_win=deliveries.groupby(['venue','winner']).apply(lambda x:x['winner'].count()).reset_index(name='Match Wins')
venue_win_pvt=pd.pivot(venue_win,values='Match Wins',index='venue',columns='winner')
venue_win_pvt.replace(np.NaN,0)
plt.figure(figsize=(20,10))
htmp=sns.heatmap(venue_win_pvt,annot=True,fmt='g',cmap='PuBuGn')
plt.xlabel('Teams',size=25)
plt.ylabel('Venues',size=25)
plt.title('Team wise wins at the Venues',size=45)
