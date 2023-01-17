import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")
ten_seasons_data = pd.read_csv('../input/english-premier-league-data-2009-2019/English_Premier_League_data_2009-2019.csv')

decade_data = ten_seasons_data.copy()

decade_data.head()
decade_data.describe()
decade_data.isnull().sum()
decade_data.dtypes
decade_data['Date'] = pd.to_datetime(decade_data['Date'], format="%Y/%m/%d")
decade_data = decade_data.drop(['Div'], axis=1) #dropping first column 'Div' as value is always E0



decade_data.columns = ['Date','HomeTeam','AwayTeam','FT_Home_Goal','FT_Away_Goal','FT_Result','HT_Home_Goal','HT_Away_Goal',

                        'HT_Result','Referee','H_Shots','A_Shots','H_Shots_Target','A_Shots_Target','H_Foul',

                        'A_Foul','H_Corner','A_Corner','H_Yellow','A_Yellow','H_Red','A_Red']



decade_data.head()
total_matches = decade_data['Date'].count()

print('Total matches played during the 10 seasons is : ' +str(total_matches))
all_teams = decade_data['HomeTeam'].unique()

all_teams_count = decade_data['HomeTeam'].nunique()

print('Total teams which played in the EPL during the ten seasons : '+str(all_teams_count))

print('\n')

print('The teams are : \n'+str(all_teams))
total_games_each_team = decade_data['HomeTeam'].value_counts() + decade_data['AwayTeam'].value_counts()

each_team_games = pd.DataFrame(total_games_each_team).sort_index(axis = 0) 

each_team_games.columns = ['Total Games']

each_team_games
each_team_games.plot(kind='bar',color='brown', legend=False, figsize=(20,10))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('Teams',fontsize=20)

plt.ylabel('Total games (EPL) in seasons 2009 - 2019',fontsize=20)

plt.title('Total games per team during 2009 - 2019',fontsize=20, color='red')

plt.show()
ten_season_teams = each_team_games[each_team_games['Total Games']==380]

ten_season_teams
one_season_teams = each_team_games[each_team_games['Total Games']==38]

one_season_teams
all_matches_results = pd.DataFrame(decade_data['FT_Result'].value_counts())

all_matches_results
labels = ['Home Team Wins','Away Team Wins','Draw']

all_matches_results.plot(kind='pie', y = 'FT_Result', autopct='%1.1f%%', 

 startangle=180, shadow=False, labels=labels, legend = False, fontsize=14, figsize=(5,5))

plt.title('Percentage share of match results',fontsize=20, color='red')
referees_count = decade_data['Referee'].nunique()

print('Number of referees who officiated the EPL matches between 2009 and 2019 : '+str(referees_count))

all_referees = pd.DataFrame(decade_data['Referee'].value_counts()).sort_index(axis = 0)

all_referees
all_referees.plot(kind='barh',color='orange', legend=False, figsize=(15,15))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.ylabel('Referees',fontsize=14)

plt.xlabel('Total games officiated in seasons 2009 - 2019',fontsize=14)

plt.title('Number of games officiated by referees during 2009 - 2019',fontsize=20, color='red')

plt.show()
come_backs = decade_data[((decade_data['HT_Result']=='A') & (decade_data['FT_Result']=='H'))

                         | 

                         ((decade_data['HT_Result']=='H') & (decade_data['FT_Result']=='A'))]



come_back_wins = come_backs.shape[0]



print('Number of games in which teams loosing at half time come back to win the game at full time : '+str(come_back_wins))
#come_backs

come_backs.head()
come_backs_year_sort = pd.DataFrame()

come_backs_year_sort['comeback_wins_per_year'] = come_backs['FT_Result'].groupby([come_backs.Date.dt.year]).agg('count')

come_backs_year_sort
season_start=9

season_end=10

season_list = []



for x in range (10):

    for y in range (380):

        season_list.append(('0'+str(season_start)+'-'+str(season_end))[-5:]) # the value '0' is added to make 9 as 09.

    season_start = season_start + 1

    season_end = season_end + 1
season_df = pd.DataFrame({'Season':season_list})



decade_data_by_seasons = pd.concat([season_df,decade_data], axis=1)

decade_data_by_seasons.head()
come_backs_updated = decade_data_by_seasons[

                        ((decade_data_by_seasons['HT_Result']=='A') & (decade_data_by_seasons['FT_Result']=='H'))

                         | 

                         ((decade_data_by_seasons['HT_Result']=='H') & (decade_data_by_seasons['FT_Result']=='A'))]





come_backs_season_sort = pd.DataFrame()

come_backs_season_sort['comeback_wins_per_season'] = come_backs['FT_Result'].groupby([come_backs_updated.Season]).agg('count')

come_backs_season_sort
def value_and_percentage(x): 

    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)





plt.figure(figsize=(9,9))

values = come_backs_season_sort['comeback_wins_per_season']

labels = decade_data_by_seasons['Season'].unique()

total = np.sum(values)

colors = ['#8BC34A','Pink','#FE7043','Turquoise','#D4E157','Grey','#EAB300','#AA7043','Violet','Orange']

plt.pie (values , labels= labels , colors= colors , 

         startangle=45 , autopct=value_and_percentage, pctdistance=0.85, 

         explode=[0,0,0,0.1,0,0,0,0,0,0] )

my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Comeback wins split among seasons',fontsize=20, color='red')

plt.show()
home_team_wins = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='H']

                  .groupby([decade_data_by_seasons.HomeTeam]).agg('count'))[['FT_Result']]

home_team_wins.columns = ['Home_Wins']

home_team_wins.index.names = ['Team']
home_team_loss = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='A']

                  .groupby([decade_data_by_seasons.HomeTeam]).agg('count'))[['FT_Result']]

home_team_loss.columns = ['Home_Loss']

home_team_loss.index.names = ['Team']
away_team_wins = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='A']

                  .groupby([decade_data_by_seasons.AwayTeam]).agg('count'))[['FT_Result']]

away_team_wins.columns = ['Away_Wins']

away_team_wins.index.names = ['Team']
away_team_loss = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='H']

                  .groupby([decade_data_by_seasons.AwayTeam]).agg('count'))[['FT_Result']]

away_team_loss.columns = ['Away_Loss']

away_team_loss.index.names = ['Team']
home_team_draw = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='D']

                  .groupby([decade_data_by_seasons.HomeTeam]).agg('count'))[['FT_Result']]

home_team_draw.columns = ['Home_Draw']

home_team_draw.index.names = ['Team']





away_team_draw = (decade_data_by_seasons[decade_data_by_seasons['FT_Result']=='D']

                  .groupby([decade_data_by_seasons.AwayTeam]).agg('count'))[['FT_Result']]

away_team_draw.columns = ['Away_Draw']

away_team_draw.index.names = ['Team']





total_draw_matches = home_team_draw['Home_Draw'] + away_team_draw['Away_Draw']

home_and_away_draws = pd.DataFrame(total_draw_matches)

home_and_away_draws.columns = ['Draws-Home_and_Away']
total_wins = home_team_wins['Home_Wins'] + away_team_wins['Away_Wins']

home_and_away_wins = pd.DataFrame(total_wins)

home_and_away_wins.columns = ['Wins-Home_and_Away']



total_points_decade = (home_and_away_wins['Wins-Home_and_Away'] * 3 ) + (home_and_away_draws['Draws-Home_and_Away'])



ten_season_points = pd.DataFrame(total_points_decade)

ten_season_points.columns = ['Total_points_in_decade']
teams_stats_table = pd.concat([each_team_games, home_team_wins, home_team_loss, away_team_wins, 

                               away_team_loss, home_and_away_draws, ten_season_points], axis=1)

teams_stats_table
allseason_teams = teams_stats_table[teams_stats_table.index.isin(ten_season_teams.index)]

allseason_teams_points = pd.DataFrame(allseason_teams['Total_points_in_decade'])

allseason_teams_points
plt.rcParams["figure.figsize"] = (15,10)

plt.bar(allseason_teams_points.index, allseason_teams_points['Total_points_in_decade'], 

        color=plt.cm.Paired((np.arange(len(allseason_teams_points)))),width = 0.5)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('Teams which played all seasons during 2009 - 2019',fontsize=20)

plt.ylabel('Total points collected in all seasons during 2009 - 2019',fontsize=20)

plt.title('Total points collected by teams which played in all seasons during 2009 - 2019',fontsize=20, color='red')



x_cord = -0.1

for z in range (len(allseason_teams_points)):

    plt.text(x_cord, allseason_teams_points.Total_points_in_decade[z] + 10, allseason_teams_points.Total_points_in_decade[z])

    x_cord = x_cord + 1

    

plt.show()
max_goals_per_game = (decade_data_by_seasons['FT_Home_Goal'] + decade_data_by_seasons['FT_Away_Goal']).max()



print('Maximum number of goals scored in a single game is : '+str(max_goals_per_game))
decade_data_by_seasons[(decade_data_by_seasons['FT_Home_Goal'] + decade_data_by_seasons['FT_Away_Goal']) == max_goals_per_game]
each_team_home_goals = decade_data_by_seasons['FT_Home_Goal'].groupby(decade_data_by_seasons['HomeTeam']).sum()



each_team_away_goals = decade_data_by_seasons['FT_Away_Goal'].groupby(decade_data_by_seasons['AwayTeam']).sum()



each_team_goal_stats = pd.DataFrame(index = each_team_games.index)

each_team_goal_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_goals), pd.DataFrame(each_team_away_goals)

                                , pd.DataFrame(each_team_home_goals + each_team_away_goals) ], axis=1)



each_team_goal_stats.columns = ['Total Games', 'HomeGoals', 'AwayGoals', 'TotalGoals']



each_team_goal_stats['AvgGoals_homeGame'] = (each_team_goal_stats['HomeGoals']/(each_team_goal_stats['Total Games']/2)).round(2)

each_team_goal_stats['AvgGoals_awayGame'] = (each_team_goal_stats['AwayGoals']/(each_team_goal_stats['Total Games']/2)).round(2)



each_team_goal_stats
each_team_goal_stats.plot(y=["AvgGoals_homeGame", "AvgGoals_awayGame"], kind="barh", legend=True, figsize=(15,15))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.ylabel('Teams',fontsize=20)

plt.xlabel('Average number of goals per game',fontsize=20)

plt.title('Average number of goals scored per game during 2009 - 2019',fontsize=20, color='red') 

plt.show()
dominant_performances_home = decade_data_by_seasons[(decade_data_by_seasons['FT_Home_Goal'] - 

                                                     decade_data_by_seasons['FT_Away_Goal'] >= 3)

                                                   ].groupby([decade_data_by_seasons.HomeTeam]).agg('count')[['HomeTeam']]





dominant_performances_away = decade_data_by_seasons[(decade_data_by_seasons['FT_Away_Goal'] - 

                                                     decade_data_by_seasons['FT_Home_Goal'] >= 3)

                                                   ].groupby([decade_data_by_seasons.AwayTeam]).agg('count')[['AwayTeam']]



#since some teams have dominant performances only at home or away, we use merge by using index from both dataframes

dominant_performances = pd.merge(dominant_performances_home, dominant_performances_away, how = 'outer', 

                                 left_index=True, right_index=True)



dominant_performances.fillna(0, inplace = True)



dominant_performances['total_dominant_performances'] = dominant_performances['HomeTeam'] + dominant_performances['AwayTeam']



dominant_performances = dominant_performances.astype('int64')

dominant_performances
others_home_dominant_games = dominant_performances[dominant_performances.HomeTeam < 10].sum()['HomeTeam']



others_away_dominant_games = dominant_performances[dominant_performances.AwayTeam < 10].sum()['AwayTeam']
dominant_home_games = pd.DataFrame(dominant_performances.HomeTeam)

others_home = pd.Series({'HomeTeam':others_home_dominant_games},name='Others')

dominant_home_games = dominant_home_games.append(others_home)

dominant_home_games = dominant_home_games[dominant_home_games.HomeTeam >= 10]

dominant_home_games
def value_and_percentage(x): 

    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)





plt.figure(figsize=(9,9))

values = dominant_home_games['HomeTeam']

labels = dominant_home_games.index

total = np.sum(values)

colors = ['#8BC34A','Pink','Olive','Grey','#FE7043','Turquoise',

          '#EAB300','Violet','Orange','Gold','Skyblue','#D4E157','#AA7043']

plt.pie (values , labels= labels , colors= colors , 

         startangle=45 , autopct=value_and_percentage, pctdistance=0.85)

my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Victories with a goal margin of 3 or more at home',fontsize=20, color='red')

plt.show()
dominant_away_games = pd.DataFrame(dominant_performances.AwayTeam)

others_away = pd.Series({'AwayTeam':others_away_dominant_games},name='Others')

dominant_away_games = dominant_away_games.append(others_away)

dominant_away_games = dominant_away_games[dominant_away_games.AwayTeam >= 10]

dominant_away_games
def value_and_percentage(x): 

    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)





plt.figure(figsize=(9,9))

values = dominant_away_games['AwayTeam']

labels = dominant_away_games.index

total = np.sum(values)

colors = ['#8BC34A','Pink','#FE7043','Turquoise','#EAB300','#D4E157','#AA7043']

plt.pie (values , labels= labels , colors= colors , 

         startangle=45 , autopct=value_and_percentage, pctdistance=0.85)

my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Victories with a goal margin of 3 or more away from home',fontsize=20, color='red')

plt.show()
season_home_wins = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'H']

                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).agg('count')

                                .unstack().fillna(0).stack()['FT_Result']).reset_index()



season_home_wins.columns = ['Season', 'Team', 'H_Wins']





season_away_wins = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'A']

                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).agg('count')

                                .unstack().fillna(0).stack()['FT_Result']).reset_index()



season_away_wins.columns = ['Season', 'Team', 'A_Wins']





season_home_draws = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'D']

                                 .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).agg('count')

                                 .unstack().fillna(0).stack()['FT_Result']).reset_index()



season_home_draws.columns = ['Season', 'Team', 'H_Draws']





season_away_draws = pd.DataFrame( decade_data_by_seasons[decade_data_by_seasons['FT_Result'] == 'D']

                                 .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).agg('count')

                                 .unstack().fillna(0).stack()['FT_Result']).reset_index()



season_away_draws.columns = ['Season', 'Team', 'A_Draws']
season_points_per_team = pd.DataFrame(season_home_wins['Team'])



season_points_per_team = pd.concat([season_points_per_team, pd.DataFrame(season_home_wins.H_Wins), 

                                    pd.DataFrame(season_away_wins.A_Wins), pd.DataFrame(season_home_draws.H_Draws), 

                                    pd.DataFrame(season_away_draws.A_Draws)], axis=1)



season_points_per_team = season_points_per_team.set_index(season_home_wins.Season)





season_points_per_team['Points'] = 3 * (season_points_per_team.H_Wins + 

                                        season_points_per_team.A_Wins) + (season_points_per_team.H_Draws + 

                                                                          season_points_per_team.A_Draws)



season_points_per_team = season_points_per_team[season_points_per_team.Points != 0]

#season_points_per_team

season_points_per_team.head()
ten_season_teams_points = season_points_per_team[season_points_per_team['Team'].isin(ten_season_teams.index)][['Team','Points']]

ten_season_teams_points = ten_season_teams_points.reset_index()

#ten_season_teams_points

ten_season_teams_points.head()
for club in ten_season_teams_points.Team.unique() :

    plt.plot(ten_season_teams_points[ten_season_teams_points['Team'] == club]['Season'], 

             ten_season_teams_points[ten_season_teams_points['Team'] == club]['Points'],  

             marker='o', label=club)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.legend()

plt.xlabel('Season', fontsize = 20)

plt.ylabel('Points', fontsize = 20)

plt.title('Points collected by teams who played in all 10 seasons',fontsize=20, color='red')

plt.show()
each_team_home_shots = decade_data_by_seasons['H_Shots'].groupby(decade_data_by_seasons['HomeTeam']).sum()



each_team_away_shots = decade_data_by_seasons['A_Shots'].groupby(decade_data_by_seasons['AwayTeam']).sum()



each_team_home_shots_target = decade_data_by_seasons['H_Shots_Target'].groupby(decade_data_by_seasons['HomeTeam']).sum()



each_team_away_shots_target = decade_data_by_seasons['A_Shots_Target'].groupby(decade_data_by_seasons['AwayTeam']).sum()





each_team_shot_stats = pd.DataFrame(index = each_team_games.index)

each_team_shot_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_shots), pd.DataFrame(each_team_away_shots), 

                                  pd.DataFrame(each_team_home_shots + each_team_away_shots), 

                                  pd.DataFrame(each_team_home_shots_target), pd.DataFrame(each_team_away_shots_target), 

                                  pd.DataFrame(each_team_home_shots_target + each_team_away_shots_target)], axis=1)



each_team_shot_stats.columns = ['Total Games', 'HomeShots', 'AwayShots', 'TotalShots', 

                               'HomeShotsTarget', 'AwayShotsTarget', 'TotalShotsTarget']



each_team_shot_stats['AvgShots_homeGame'] = (each_team_shot_stats['HomeShots']/(each_team_shot_stats['Total Games']/2)).round(2)

each_team_shot_stats['AvgShots_awayGame'] = (each_team_shot_stats['AwayShots']/(each_team_shot_stats['Total Games']/2)).round(2)



each_team_shot_stats['AvgShotstarget_homeGame'] = (each_team_shot_stats['HomeShotsTarget']/

                                                   (each_team_shot_stats['Total Games']/2)).round(2)

each_team_shot_stats['AvgShotstarget_awayGame'] = (each_team_shot_stats['AwayShotsTarget']/

                                                   (each_team_shot_stats['Total Games']/2)).round(2)

each_team_shot_and_goals = pd.concat([each_team_shot_stats, each_team_goal_stats], axis=1)

each_team_shot_and_goals = each_team_shot_and_goals.loc[:,~each_team_shot_and_goals.columns.duplicated()]



#reorder columns

each_team_shot_and_goals = each_team_shot_and_goals[['Total Games', 'HomeShots', 'HomeShotsTarget','HomeGoals',

                                                     'AwayShots', 'AwayShotsTarget', 'AwayGoals', 

                                                     'TotalShots', 'TotalShotsTarget', 'TotalGoals', 

                                                     'AvgShots_homeGame', 'AvgShotstarget_homeGame', 'AvgGoals_homeGame', 

                                                    'AvgShots_awayGame', 'AvgShotstarget_awayGame', 'AvgGoals_awayGame']]



#rename columns to fit

each_team_shot_and_goals.columns = ['Games', 'HS', 'HST', 'HG','AS', 'AST', 'AG', 

                                    'TS', 'TST', 'TG', 

                                    'AvgSHG', 'AvgSTHG', 'AvgGHG', 'AvgSAG', 'AvgSTAG', 'AvgGAG']



each_team_shot_and_goals
ten_season_teams_shots_and_goals = each_team_shot_and_goals[each_team_shot_and_goals.index.isin(ten_season_teams.index)]

ten_season_teams_shots_and_goals
ten_season_teams_shots_and_goals.plot(y=['TG', 'TST','TS'], kind='bar', 

                                      label=['Total Goals', 'Total Shots on Target', 'Total Shots'], 

                                      color=['#8BC34A','#EAB300','tomato'])

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('Team', fontsize = 20)

plt.ylabel('Shots', fontsize = 20)



plt.figtext(0.25, 0.93, "Shots and Goals stats of teams who played all 10 seasons", 

            fontsize=20, color='Black')



plt.figtext(0.26, 0.91, "Total Goals", fontsize=20, color='#8BC34A', ha ='left', va='top')

plt.figtext(0.38, 0.91, "vs", fontsize=20, color='Black', ha ='left', va='top')

plt.figtext(0.52, 0.91, "Total Shots on Target", fontsize=20, color='#EAB300', ha ='center', va='top')

plt.figtext(0.65, 0.91, "vs", fontsize=20, color='Black', ha ='center', va='top')

plt.figtext(0.78, 0.91, "Total Shots", fontsize=20, color='tomato', ha ='right', va='top')



plt.show()
ten_season_teams_shots_and_goals.plot(y=['AvgGHG', 'AvgSTHG','AvgSHG'], kind='bar', 

                                      label=['Avg Goals/Home game', 'Avg Shots on Target/Home Game', 'Avg Shots/Home Game'], 

                                      color=['#8BC34A','#EAB300','tomato'])

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('Team', fontsize = 20)

plt.ylabel('Shots Avg', fontsize = 20)

plt.ylim(0, 20)



plt.figtext(0.15, 0.93, "Shots and Goals Averages/Game @Home of teams who played all 10 seasons", 

            fontsize=20, color='Black')



plt.figtext(0.11, 0.91, "Avg Goals/Home game", fontsize=20, color='#8BC34A', ha ='left', va='top')

plt.figtext(0.33, 0.91, "vs", fontsize=20, color='Black', ha ='left', va='top')

plt.figtext(0.51, 0.91, "Avg Shots on Target/Home Game", fontsize=20, color='#EAB300', ha ='center', va='top')

plt.figtext(0.68, 0.91, "vs", fontsize=20, color='Black', ha ='center', va='top')

plt.figtext(0.91, 0.91, "Avg Shots/Home Game", fontsize=20, color='tomato', ha ='right', va='top')



plt.show()
ten_season_teams_shots_and_goals.plot(y=['AvgGAG', 'AvgSTAG','AvgSAG'], kind='bar', 

                                      label=['Avg Goals/Away game', 'Avg Shots on Target/Away Game', 'Avg Shots/Away Game'], 

                                      color=['#8BC34A','#EAB300','tomato'])

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('Team', fontsize = 20)

plt.ylabel('Shots Avg', fontsize = 20)

plt.ylim(0, 16)



plt.figtext(0.15, 0.93, "Shots and Goals Averages/Game @Away of teams who played all 10 seasons", 

            fontsize=20, color='Black')



plt.figtext(0.11, 0.91, "Avg Goals/Away game", fontsize=20, color='#8BC34A', ha ='left', va='top')

plt.figtext(0.33, 0.91, "vs", fontsize=20, color='Black', ha ='left', va='top')

plt.figtext(0.51, 0.91, "Avg Shots on Target/Away Game", fontsize=20, color='#EAB300', ha ='center', va='top')

plt.figtext(0.68, 0.91, "vs", fontsize=20, color='Black', ha ='center', va='top')

plt.figtext(0.91, 0.91, "Avg Shots/Away Game", fontsize=20, color='tomato', ha ='right', va='top')



plt.show()
#CRH - coversion rate @ Home

#CRA - convertio rate @ Away



#two diff methods to divide columns

ten_season_teams_shots_and_goals['CRH%'] = (ten_season_teams_shots_and_goals['HG']/

                                           ten_season_teams_shots_and_goals['HS']*100).round(2)



ten_season_teams_shots_and_goals.loc[:,'CRA%'] = (ten_season_teams_shots_and_goals.loc[:,'AG']/

                                                 ten_season_teams_shots_and_goals.loc[:,'AS']*100).round(2)



ten_season_teams_shots_and_goals
ten_season_teams_shots_and_goals.plot(y=['CRH%', 'CRA%'], kind='bar', 

                                      label=['Conversion rate in Home Matches', 'Conversion rate in Away Mathes'], 

                                      color=['Turquoise','Violet'])

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('Team', fontsize = 20)

plt.ylabel('Converison rate in %', fontsize = 20)



plt.title('Conversion rates (%) at home and away matches', fontsize = 20, color='red')



plt.show()
each_team_home_goals_conceded = decade_data_by_seasons['FT_Away_Goal'].groupby(decade_data_by_seasons['HomeTeam']).sum()



each_team_away_goals_conceded = decade_data_by_seasons['FT_Home_Goal'].groupby(decade_data_by_seasons['AwayTeam']).sum()



each_team_goal_conceded_stats = pd.DataFrame(index = each_team_games.index)

each_team_goal_conceded_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_goals_conceded), 

                                           pd.DataFrame(each_team_away_goals_conceded), 

                                           pd.DataFrame(each_team_home_goals_conceded + each_team_away_goals_conceded) ], 

                                          axis=1)



each_team_goal_conceded_stats.columns = ['Total Games', 'GoalsConcededHome', 'GoalsConcededAway', 'TotalGoalsConceded']



each_team_goal_conceded_stats['AvgGoalsConceded_homeGame'] = (each_team_goal_conceded_stats['GoalsConcededHome']/

                                                              (each_team_goal_conceded_stats['Total Games']/2)).round(2)



each_team_goal_conceded_stats['AvgGoalsConceded_awayGame'] = (each_team_goal_conceded_stats['GoalsConcededAway']/

                                                              (each_team_goal_conceded_stats['Total Games']/2)).round(2)



each_team_goal_conceded_stats
each_team_goal_conceded_stats.plot(y=["AvgGoalsConceded_homeGame", "AvgGoalsConceded_awayGame"], 

                                   kind="barh", legend=True, figsize=(15,15))

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.ylabel('Teams',fontsize=20)

plt.xlabel('Average number of goals conceded per game',fontsize=20)

plt.title('Average number of goals conceded per game during 2009 - 2019',fontsize=20, color='red') 

plt.show()
total_yellows_home = decade_data_by_seasons['H_Yellow'].sum()

total_reds_home = decade_data_by_seasons['H_Red'].sum()

total_yellows_away = decade_data_by_seasons['A_Yellow'].sum()

total_reds_away = decade_data_by_seasons['A_Red'].sum()



yellow_and_red_cards = [total_yellows_home, total_reds_home, total_yellows_away, total_reds_away]



print('Total yellow cards awarded to home side : ' + str(yellow_and_red_cards[0]))

print('Total red cards awarded to home side : ' + str(yellow_and_red_cards[1]))

print('Total yellow cards awarded to away side : ' + str(yellow_and_red_cards[2]))

print('Total red cards awarded to away side : ' + str(yellow_and_red_cards[3]))
def value_and_percentage(x): 

    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)





plt.figure(figsize=(9,9))

values = yellow_and_red_cards

labels = ['Yellow cards for home team', 'Red cards for home team', 'Yellow cards for away team', 'Red cards for away team']

total = np.sum(values)

colors = ['Gold','#FE7043','Turquoise','Violet']

plt.pie (values , labels= labels , colors= colors , 

         startangle=45 , autopct=value_and_percentage, pctdistance=0.85, 

         textprops={'fontsize': 14}, explode=[0.02,0,0.02,0] )



my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Yellow and red cards distribution',fontsize=20, color='red')

plt.show()
each_team_home_yellows = decade_data_by_seasons['H_Yellow'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_home_reds = decade_data_by_seasons['H_Red'].groupby(decade_data_by_seasons['HomeTeam']).sum()



each_team_away_yellows = decade_data_by_seasons['A_Yellow'].groupby(decade_data_by_seasons['AwayTeam']).sum()

each_team_away_reds = decade_data_by_seasons['A_Red'].groupby(decade_data_by_seasons['AwayTeam']).sum()





each_team_card_stats = pd.DataFrame(index = each_team_games.index)

each_team_card_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_yellows), pd.DataFrame(each_team_home_reds),

                                  pd.DataFrame(each_team_away_yellows), pd.DataFrame(each_team_away_reds), 

                                  pd.DataFrame(each_team_home_yellows + each_team_away_yellows), 

                                  pd.DataFrame(each_team_home_reds + each_team_away_reds)], axis=1)



each_team_card_stats.columns = ['Total Games', 'HomeYellows', 'HomeReds', 'AwayYellows', 

                                'AwayReds', 'TotalYellows', 'TotalReds']



#HGY - HomeGameYellow; HGR - HomeGameRed; AGY - AwayGameYellow; AGR - AwayGameRed

each_team_card_stats['AvgHGY'] = (each_team_card_stats['HomeYellows']/(each_team_card_stats['Total Games']/2)).round(2)

each_team_card_stats['AvgHGR'] = (each_team_card_stats['HomeReds']/(each_team_card_stats['Total Games']/2)).round(2)

each_team_card_stats['AvgAGY'] = (each_team_card_stats['AwayYellows']/(each_team_card_stats['Total Games']/2)).round(2)

each_team_card_stats['AvgAGR'] = (each_team_card_stats['AwayReds']/(each_team_card_stats['Total Games']/2)).round(2)



each_team_card_stats
ax1 = each_team_card_stats.plot( y=["TotalYellows"], kind="barh",

                          legend=False, color =('gold'), figsize=(15,15),

                          title='Yellow cards collected @home/away (combined) during 2009 - 2019', fontsize=14)

ax1.set(xlabel='Number of yellow cards', ylabel='Team')

ax1.title.set_size(20)

ax1yaxis_label = ax1.yaxis.get_label()

ax1yaxis_label.set_fontsize(14)

ax1xaxis_label = ax1.xaxis.get_label()

ax1xaxis_label.set_fontsize(14)

plt.show()
ax2 = each_team_card_stats.plot( y=["TotalReds"], kind="barh", 

                          legend=False, color =('red'), figsize=(15,15), 

                          title='Red cards collected @home/away (combined) during 2009 - 2019', fontsize=14)

ax2.set(xlabel='Number of cards', ylabel='Team')

ax2.title.set_size(20)

ax2yaxis_label = ax2.yaxis.get_label()

ax2yaxis_label.set_fontsize(14)

ax2xaxis_label = ax2.xaxis.get_label()

ax2xaxis_label.set_fontsize(14)



plt.show()
ax1 = each_team_card_stats.plot( y=["AvgHGY", "AvgAGY"], kind="bar",

                          legend=False, color =('olive','darkorange'), figsize=(40,10),

                          title='Average number of Yellow card collected per home/away game during 2009 - 2019', fontsize=30)

ax1.set(xlabel='', ylabel='Average number of cards per game\n') #we dont give x label here. Both plots will have same x axis.

ax1.title.set_size(30)

ax1yaxis_label = ax1.yaxis.get_label()

ax1yaxis_label.set_fontsize(30)



ax2 = each_team_card_stats.plot( y=["AvgHGR", "AvgAGR"], kind="bar", 

                          legend=False, color =('blue','red'), figsize=(40,10), 

                          title='Average number of Red card collected per home/away game during 2009 - 2019', fontsize=30)

ax2.set(xlabel='Team', ylabel='Average number of cards per game\n')

ax2.title.set_size(30)

ax2yaxis_label = ax2.yaxis.get_label()

ax2yaxis_label.set_fontsize(30)

ax2xaxis_label = ax2.xaxis.get_label()

ax2xaxis_label.set_fontsize(30)



plt.show()
referee_home_yellows = decade_data_by_seasons['H_Yellow'].groupby(decade_data_by_seasons['Referee']).sum()

referee_home_reds = decade_data_by_seasons['H_Red'].groupby(decade_data_by_seasons['Referee']).sum()

referee_away_yellows = decade_data_by_seasons['A_Yellow'].groupby(decade_data_by_seasons['Referee']).sum()

referee_away_reds = decade_data_by_seasons['A_Red'].groupby(decade_data_by_seasons['Referee']).sum()



each_referee_card_stats = pd.DataFrame(index = all_referees.index)

each_referee_card_stats = pd.concat([all_referees, pd.DataFrame(referee_home_yellows), pd.DataFrame(referee_home_reds),

                                  pd.DataFrame(referee_away_yellows), pd.DataFrame(referee_away_reds), 

                                  pd.DataFrame(referee_home_yellows + referee_away_yellows), 

                                  pd.DataFrame(referee_home_reds + referee_away_reds)], axis=1)



each_referee_card_stats.columns = ['Total Games', 'YellowsToHomeSide', 'RedsToHomeSide', 'YellowsToAwaySide', 

                                'RedsToAwaySide', 'YellowsTotal', 'RedsTotal']

each_referee_card_stats
ax1 = each_referee_card_stats.plot( y=["YellowsTotal"], kind="bar",

                          legend=False, color =('gold'), figsize=(40,10),

                          title='Yellow cards awarded during 2009 - 2019 in EPL', fontsize=30)



ax1.set(xlabel='', ylabel='Cards awarded') #we dont give x label here. Both plots will have same x axis.

ax1.title.set_size(30)

ax1yaxis_label = ax1.yaxis.get_label()

ax1yaxis_label.set_fontsize(30)



ax2 = each_referee_card_stats.plot( y=["RedsTotal"], kind="bar", 

                          legend=False, color =('crimson'), figsize=(40,10), 

                          title='Red cards awarded during 2009 - 2019 in EPL', fontsize=30)



ax2.set(xlabel='Referee', ylabel='Cards awarded')

ax2.title.set_size(30)

ax2yaxis_label = ax2.yaxis.get_label()

ax2yaxis_label.set_fontsize(30)



ax2xaxis_label = ax2.xaxis.get_label()

ax2xaxis_label.set_fontsize(30)



plt.show()
home_fouls = decade_data_by_seasons['H_Foul'].groupby(decade_data_by_seasons['HomeTeam']).sum()

away_fouls = decade_data_by_seasons['A_Foul'].groupby(decade_data_by_seasons['AwayTeam']).sum()





each_team_foul_stats = pd.DataFrame(index = each_team_games.index)

each_team_foul_stats = pd.concat([each_team_games, pd.DataFrame(home_fouls), pd.DataFrame(away_fouls), 

                                  pd.DataFrame(home_fouls + away_fouls)], axis=1)



each_team_foul_stats.columns = ['Total Games', 'FoulsCommitted@Home', 'FoulsCommitted@Away', 'TotalFoulsCommitted']





#HGF - HomeGameFoul; AGF - AwayGameFoul

each_team_foul_stats['AvgHGF'] = (each_team_foul_stats['FoulsCommitted@Home']/

                                  (each_team_foul_stats['Total Games']/2)).round(2)



each_team_foul_stats['AvgAGF'] = (each_team_foul_stats['FoulsCommitted@Away']/

                                  (each_team_foul_stats['Total Games']/2)).round(2)



each_team_foul_stats
each_team_foul_stats.plot(y=["AvgHGF", "AvgAGF"], 

                          kind="barh", legend=True, figsize=(15,15), color = ['crimson', 'darkgreen'], 

                          label=['Average Fouls in Home Game', 'Average Fouls in Away Game'])

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.ylabel('Teams',fontsize=20)

plt.xlabel('Number of fouls per game',fontsize=20)

plt.title('Average number of fouls commited per game at home and away games during 2009 - 2019',fontsize=18, color='red') 

plt.show()
each_team_fouls_to_cards = pd.DataFrame(index = each_team_games.index)

each_team_fouls_to_cards = pd.concat([each_team_games, each_team_foul_stats['TotalFoulsCommitted'], 

                                  each_team_card_stats['TotalYellows'], 

                                  each_team_card_stats['TotalReds']], axis=1)



each_team_fouls_to_cards['YellowsPerFoul %'] = ((each_team_fouls_to_cards['TotalYellows']/

                                             each_team_fouls_to_cards['TotalFoulsCommitted'])*100).round(2)



each_team_fouls_to_cards['RedsPerFoul %'] = ((each_team_fouls_to_cards['TotalReds']/

                                             each_team_fouls_to_cards['TotalFoulsCommitted'])*100).round(2)



each_team_fouls_to_cards
ax1 = each_team_fouls_to_cards.plot( y=["YellowsPerFoul %"], kind="bar",

                          legend=False, color =('gold'), figsize=(40,10),

                          title='Yellow cards per foul in % (cards per 100 fouls) for each team', fontsize=30)



ax1.set(xlabel='', ylabel='Yellow cards per foul in %\n (cards per 100 fouls) \n') #we dont give x label here. Both plots will have same x axis.

ax1.title.set_size(30)

ax1yaxis_label = ax1.yaxis.get_label()

ax1yaxis_label.set_fontsize(30)



ax2 = each_team_fouls_to_cards.plot( y=["RedsPerFoul %"], kind="bar", 

                          legend=False, color =('crimson'), figsize=(40,10), 

                          title='Red cards per foul in % (cards per 100 fouls) for each team', fontsize=30)



ax2.set(xlabel='Teams', ylabel='Red cards per foul in %\n (cards per 100 fouls) \n')

ax2.title.set_size(30)

ax2yaxis_label = ax2.yaxis.get_label()

ax2yaxis_label.set_fontsize(30)



ax2xaxis_label = ax2.xaxis.get_label()

ax2xaxis_label.set_fontsize(30)



plt.show()
each_team_home_corners_gained = decade_data_by_seasons['H_Corner'].groupby(decade_data_by_seasons['HomeTeam']).sum()

each_team_home_corners_conceded = decade_data_by_seasons['A_Corner'].groupby(decade_data_by_seasons['HomeTeam']).sum()



each_team_away_corners_gained = decade_data_by_seasons['A_Corner'].groupby(decade_data_by_seasons['AwayTeam']).sum()

each_team_away_corners_conceded = decade_data_by_seasons['H_Corner'].groupby(decade_data_by_seasons['AwayTeam']).sum()





each_team_corner_stats = pd.DataFrame(index = each_team_games.index)

each_team_corner_stats = pd.concat([each_team_games, pd.DataFrame(each_team_home_corners_gained), 

                                    pd.DataFrame(each_team_home_corners_conceded),

                                    pd.DataFrame(each_team_away_corners_gained), 

                                    pd.DataFrame(each_team_away_corners_conceded), 

                                    pd.DataFrame(each_team_home_corners_gained + each_team_away_corners_gained), 

                                    pd.DataFrame(each_team_home_corners_conceded + each_team_away_corners_conceded)], 

                                   axis=1)



#CG - Corners Gained; CC - Corners Conceded

each_team_corner_stats.columns = ['Total Games', 'CG@Home', 'CC@Home', 

                                  'CG@Away', 'CC@Away', 

                                  'TotalCG', 'TotalCC']



#HCG - HomeCornersGained; HCC - HomeCornersConceded; ACG - AwayCornersGained; ACC - AwayCornersConceded

each_team_corner_stats['AvgHCG'] = (each_team_corner_stats['CG@Home']/

                                    (each_team_corner_stats['Total Games']/2)).round(2)



each_team_corner_stats['AvgHCC'] = (each_team_corner_stats['CC@Home']/

                                    (each_team_corner_stats['Total Games']/2)).round(2)



each_team_corner_stats['AvgACG'] = (each_team_corner_stats['CG@Away']/

                                    (each_team_corner_stats['Total Games']/2)).round(2)



each_team_corner_stats['AvgACC'] = (each_team_corner_stats['CC@Away']/

                                    (each_team_corner_stats['Total Games']/2)).round(2)



each_team_corner_stats
each_team_corner_stats.plot(y=["TotalCG", "TotalCC"], 

                            kind="bar", legend=True, figsize=(40,15), color = ['green', 'red'], 

                            label=['Total Corners Gained', 'Total Corners Conceded'])

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

plt.ylabel('Corners gained/conceded between 2009 - 2019',fontsize=20)

plt.xlabel('Teams',fontsize=20)

plt.title('Total corners gained/conceded between 2009 - 2019 in EPL',fontsize=20, color='red') 

plt.show()
corner_stats_ten_season_teams = each_team_corner_stats[each_team_corner_stats.index.isin(ten_season_teams.index)]

corner_stats_ten_season_teams
corner_stats_ten_season_teams.plot(y=["AvgHCG", "AvgHCC", "AvgACG", "AvgACC"], 

                                   kind="barh", legend=True, figsize=(15,15), color = ['green', 'deeppink' , 'lime', 'red'], 

                                   label = ['Avg Corners Gained @Home per game', 'Avg Corners Conceded @Home per game', 

                                           'Avg Corners Gained @Away per game', 'Avg Corners Conceded @Away per game'])

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel('Average number of corners gained/conceded per game',fontsize=20)

plt.ylabel('Teams',fontsize=20)

plt.title('Average number of corners gained/conceded per game between 2009 - 2019 in EPL',fontsize=20, color='red') 

plt.show()
max_points_teams_per_season = season_points_per_team.groupby([season_points_per_team.

                                                    index])['Points'].transform(max) == season_points_per_team['Points']



season_points_per_team[max_points_teams_per_season]
#Find goals scored by each team per season (home + away)

season_home_goals_scored = pd.DataFrame( decade_data_by_seasons['FT_Home_Goal']

                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).sum()

                                .unstack().fillna(0).stack()).reset_index()



season_home_goals_scored.columns = ['Season', 'Team', 'HGoals_Scored']



season_away_goals_scored = pd.DataFrame( decade_data_by_seasons['FT_Away_Goal']

                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).sum()

                                .unstack().fillna(0).stack()).reset_index()



season_away_goals_scored.columns = ['Season', 'Team', 'AGoals_Scored']





season_total_goals_scored = pd.DataFrame()

season_total_goals_scored = pd.concat([season_home_goals_scored.Season, season_home_goals_scored.Team, 

                                       pd.DataFrame(season_home_goals_scored['HGoals_Scored'] + 

                                                    season_away_goals_scored['AGoals_Scored'])],

                                      axis=1)





season_total_goals_scored.columns = ['Season', 'Team', 'Total_Goals_Scored']





#Find goals conceded by each team per season (home + away)

season_home_goals_conceded = pd.DataFrame( decade_data_by_seasons['FT_Away_Goal']

                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.HomeTeam]).sum()

                                .unstack().fillna(0).stack()).reset_index()



season_home_goals_conceded.columns = ['Season', 'Team', 'HGoals_Conceded']



season_away_goals_conceded = pd.DataFrame( decade_data_by_seasons['FT_Home_Goal']

                                .groupby([decade_data_by_seasons.Season, decade_data_by_seasons.AwayTeam]).sum()

                                .unstack().fillna(0).stack()).reset_index()



season_away_goals_conceded.columns = ['Season', 'Team', 'AGoals_Conceded']





season_total_goals_conceded = pd.DataFrame()

season_total_goals_conceded = pd.concat([season_home_goals_conceded.Season, season_home_goals_conceded.Team, 

                                       pd.DataFrame(season_home_goals_conceded['HGoals_Conceded'] + 

                                                    season_away_goals_conceded['AGoals_Conceded'])],

                                      axis=1)



season_total_goals_conceded.columns = ['Season', 'Team', 'Total_Goals_Conceded']





#Make a new df with Goal Difference



season_total_gd = pd.DataFrame()

season_total_gd = pd.concat([season_total_goals_scored.Season, season_total_goals_scored.Team, 

                             season_total_goals_scored.Total_Goals_Scored, season_total_goals_conceded.Total_Goals_Conceded],

                            axis=1)



season_total_gd['Goal_Difference'] = season_total_gd['Total_Goals_Scored'] - season_total_gd['Total_Goals_Conceded']





season_total_gd = season_total_gd[(season_total_gd.Total_Goals_Scored != 0)  &  (season_total_gd.Total_Goals_Conceded != 0)]

season_total_gd.set_index('Season')

#season_total_gd

season_total_gd.head()
season_total_gd.index = season_points_per_team.index





season_points_per_team_with_goal_diff = pd.concat([season_points_per_team, season_total_gd.Total_Goals_Scored, 

                                                   season_total_gd.Total_Goals_Conceded, season_total_gd.Goal_Difference], 

                                                  axis=1, sort=False)



#season_points_per_team_with_goal_diff

season_points_per_team_with_goal_diff.head()
max_points_teams_each_season = season_points_per_team_with_goal_diff.groupby([

    season_points_per_team_with_goal_diff.index])['Points'].transform(max) == season_points_per_team_with_goal_diff['Points']





max_points_teams_per_season = season_points_per_team_with_goal_diff[max_points_teams_each_season]

max_points_teams_per_season = max_points_teams_per_season.reset_index()

max_points_teams_per_season
champions = pd.DataFrame()



for season in max_points_teams_per_season.Season.unique():

    seasonal_top_pointers = max_points_teams_per_season[max_points_teams_per_season['Season']==season]

    seasonal_top_pointers_with_gd = seasonal_top_pointers[seasonal_top_pointers['Goal_Difference']==

                                                          seasonal_top_pointers['Goal_Difference'].max()]

    champions = champions.append(seasonal_top_pointers_with_gd)



champions = champions.set_index('Season')

champions
print('The different teams to be crowned EPL champions between 2009 - 2019 are : \n' + str(champions.Team.unique()))
champions_with_trophy_count = pd.DataFrame(champions['Team'].value_counts())

champions_with_trophy_count.columns = ['Trophy_Number']

champions_with_trophy_count
team_with_max_epl_trophy = champions_with_trophy_count[champions_with_trophy_count['Trophy_Number']==

                                                          champions_with_trophy_count['Trophy_Number'].max()]

team_with_max_epl_trophy
def value_and_percentage(x): 

    return '{:.2f}%\n({:.0f})'.format(x, total*x/100)





plt.figure(figsize=(9,9))

values = champions_with_trophy_count.Trophy_Number

labels = champions_with_trophy_count.index.unique()

total = np.sum(values)

colors = ['#8BC34A','dodgerblue','#FE7043','Turquoise']

plt.pie (values , colors = colors ,  labels = labels,

         startangle=45 , autopct=value_and_percentage, pctdistance=0.85, 

         textprops={'fontsize': 14}, explode=[0.05,0,0,0] )



my_circle=plt.Circle( (0,0), 0.7, color='white') # Adding circle at the centre

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.title('Number of EPL trophies between 2009 - 2019',fontsize=20, color='red')

plt.show()