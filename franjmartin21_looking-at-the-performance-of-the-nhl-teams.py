# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



import datetime



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
game_df = pd.read_csv('../input/game.csv')

game_df.info()

game_df.head()
for column in ['date_time', 'date_time_GMT']:

    game_df[column] = pd.to_datetime(game_df[column], errors='coerce')
team_info_df = pd.read_csv('../input/team_info.csv')

game_df = game_df.assign(away_team_name=game_df['away_team_id'].apply(lambda x: team_info_df[team_info_df['team_id'] == x]['teamName'].values[0]))

game_df = game_df.assign(home_team_name=game_df['home_team_id'].apply(lambda x: team_info_df[team_info_df['team_id'] == x]['teamName'].values[0]))

game_df = game_df.drop(['away_team_id', 'home_team_id', 'venue_link', 'venue_time_zone_id', 'venue_time_zone_offset'],axis=1)

game_df.head()

game_df.info()
#First I separate in a new dataframe the information about the home team, and I rename the columns to something removing the HOME or AWAY words from them.

games_home_df = game_df[['game_id', 'date_time_GMT', 'home_team_name', 'home_goals', 'away_goals', 'type', 'season']].rename(columns={'home_team_name': 'team_name', 'home_goals': 'goals_scored', 'away_goals':'goals_conceded'})

#And I add another column called 'ground' so we still now that this team played at home

games_home_df = games_home_df.assign(ground='HOME')

#Same thing for AWAY team

games_away_df = game_df[['game_id', 'date_time_GMT', 'away_team_name', 'away_goals', 'home_goals', 'type', 'season']].rename(columns={'away_team_name': 'team_name', 'away_goals': 'goals_scored', 'home_goals':'goals_conceded'})

games_away_df = games_away_df.assign(ground='AWAY')

#We have now to different datasets with the same column names. One containing all teams playing at HOME and another with all teams playing AWAY, we will concat them to make it just one dataset

games_team_df = pd.concat([games_home_df, games_away_df])

#Finally I will add another categorical column showing if the team won or lost the game

games_team_df = games_team_df.assign(outcome=games_team_df.apply(lambda x: 'WIN' if (x['goals_scored']>x['goals_conceded']) else 'LOSE', axis=1))

games_team_df.head()
#As we want to print by team, let's group the dataset first by team_name

games_team_groupby_team_df = games_team_df.groupby('team_name')

#Then, We want to print the list of teams sorted by number of games won

teams_count_games_serie = games_team_groupby_team_df['game_id'].count().sort_values(ascending=False)

plt.figure(figsize=(16,6))

ax = sns.barplot(x=teams_count_games_serie.index, y=teams_count_games_serie.values, palette='viridis')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

ax.set(ylabel='Games Played', xlabel="Team")

ax.plot()
#We filter by outcome== 'WIN', group again by team_name and then we print the 10 most succesfull teams and we plot the whole set of teams

game_win_groupby_team=games_team_df[games_team_df['outcome'] == 'WIN'].groupby('team_name')

teams_win_games_serie = game_win_groupby_team['game_id'].count().sort_values(ascending=False)

teams_win_games_serie.head(10)
plt.figure(figsize=(16,6))

ax = sns.barplot(x=teams_win_games_serie.index, y=teams_win_games_serie.values, palette='viridis')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

ax.set(ylabel='Games Win', xlabel="Team")

ax.plot()
#For this, we are going to use the two series that we used to print above charts. We start by converting both to df and renaming the game_id to something that will allow us to differentiate them later

teams_count_games_df = teams_count_games_serie.to_frame().rename(columns={'game_id': 'num_games_played'})

teams_win_games_df = teams_win_games_serie.to_frame().rename(columns={'game_id': 'num_games_won'})

#We join them. Join will use the value in the index to join both frames

teams_count_win_games_df=teams_count_games_df.join(teams_win_games_df)

#Then we compute the percentage of wins per game and sort by that new column

teams_percentage_wins_df = teams_count_win_games_df.assign(perc_wins=teams_count_win_games_df.apply(lambda x: x['num_games_won']/x['num_games_played'], axis=1)).sort_values(by='perc_wins', ascending=False)

teams_percentage_wins_df.head(10)
# We print the result

plt.figure(figsize=(16,10))

ax = sns.barplot(x='perc_wins', y='team_name', data=teams_percentage_wins_df.reset_index(), palette='viridis')

ax.set(ylabel='Teams', xlabel="% Wins")

ax.plot()
#For this we want to compute a new column, the difference of goals on each game 

games_team_diff_goals_df = games_team_df.assign(diff_goals=games_team_df['goals_scored'] - games_team_df['goals_conceded'])

#Then we group by team again and we do the addition of the new 'diff_goals' column

games_team_diff_goals_serie = games_team_diff_goals_df.groupby('team_name')['diff_goals'].sum().sort_values(ascending=False)
## Let's print the head and tail of the new serie

games_team_diff_goals_serie.head()
games_team_diff_goals_serie.tail()
plt.figure(figsize=(16,10))

ax = sns.barplot(x=games_team_diff_goals_serie.values, y=games_team_diff_goals_serie.index, palette='viridis')

ax.set(ylabel='Goal Difference', xlabel='Goal Difference')
#I create a dataframe filtering by outcome=='WIN' and grouping by team_name and gound

games_home_away_win_df = games_team_df[games_team_df['outcome']=='WIN'].groupby(['team_name', 'ground']).count()['game_id'].unstack().rename(columns={'AWAY':'win_away', 'HOME': 'win_home'})

#I create a dataframe like the one before but this time to get the total games played at home and away

games_home_away_count_df = games_team_df.groupby(['team_name', 'ground']).count()['game_id'].unstack().rename(columns={'AWAY':'count_away', 'HOME': 'count_home'})

#Then, I join both dataframes, and I compute the percentage of wins home and away per team

games_home_away_df = games_home_away_win_df.join(games_home_away_count_df)

games_home_away_df['perc_win_home'] = games_home_away_df['win_home']/games_home_away_df['count_home']

games_home_away_df['perc_win_away'] = games_home_away_df['win_away']/games_home_away_df['count_away']

#Let's print our new dataframe

games_home_away_df.head(10)
f, (ax1, ax2) = plt.subplots(1, 2)

f.set_figwidth(25)

f.set_figheight(10)

sns.barplot(x='perc_win_home', y='team_name', data=games_home_away_df.sort_values('perc_win_home', ascending=False).reset_index().head(30), palette='viridis', ax=ax1)

sns.barplot(x='perc_win_away', y='team_name', data=games_home_away_df.sort_values('perc_win_away', ascending=False).reset_index().head(30), palette='viridis', ax=ax2)

ax1.plot()

ax2.plot()
games_team_df = games_team_df.loc[game_df['season']==20172018]
def assign_outcome(x):

    outcome = ""

    if(x['goals_scored'] > x['goals_conceded']):

        outcome = outcome + 'WIN' 

    else:

        outcome = outcome + 'LOSE' 

    

    if('OT' in x['outcome'] or 'SO' in x['outcome']):

        outcome = outcome + ' OT' 

        

    return outcome



def assign_points(x):

    if(x['outcome'] == 'LOSE OT'):

        return 1

    elif(x['outcome'] == 'WIN' or x['outcome'] == 'WIN OT'):

        return 2

    else:

        return 0



games_team_df = games_team_df.assign(outcome=games_team_df.apply(assign_outcome, axis=1))

games_team_df = games_team_df.assign(points=games_team_df.apply(assign_points, axis=1))

games_team_df.head(10)
teams_division_dict = {

'Flyers':'Metropolitan',

'Devils':'Metropolitan',

'Kings':'Pacific',

'Bruins':'Atlantic',

'Lightning':'Atlantic',

'Rangers':'Metropolitan',

'Penguins':'Metropolitan',

'Sharks':'Pacific',

'Red Wings':'Atlantic',

'Canucks':'Pacific',

'Predators':'Central',

'Blackhawks':'Central',

'Canadiens':'Atlantic',

'Senators':'Atlantic',

'Wild':'Central',

'Capitals':'Metropolitan',

'Blues':'Central',

'Ducks':'Pacific',

'Coyotes':'Pacific',

'Islanders':'Metropolitan',

'Maple Leafs':'Atlantic',

'Panthers':'Atlantic',

'Sabres':'Atlantic',

'Flames':'Pacific',

'Avalanche':'Central',

'Stars':'Central',

'Blue Jackets':'Metropolitan',

'Jets':'Central',

'Oilers':'Pacific',

'Golden Knights':'Pacific',

'Hurricanes':'Metropolitan'}

games_team_df['division'] = games_team_df['team_name'].map(teams_division_dict)
teams_conference_dict={

'Flyers':'Eastern',

'Devils':'Eastern',

'Kings':'Western',

'Bruins':'Eastern',

'Lightning':'Eastern',

'Rangers':'Eastern',

'Penguins':'Eastern',

'Sharks':'Western',

'Red Wings':'Eastern',

'Canucks':'Western',

'Predators':'Western',

'Blackhawks':'Western',

'Canadiens':'Eastern',

'Senators':'Eastern',

'Wild':'Western',

'Capitals':'Eastern',

'Blues':'Western',

'Ducks':'Western',

'Coyotes':'Western',

'Islanders':'Eastern',

'Maple Leafs':'Eastern',

'Panthers':'Eastern',

'Sabres':'Eastern',

'Flames':'Western',

'Avalanche':'Western',

'Stars':'Western',

'Blue Jackets':'Eastern',

'Jets':'Western',

'Oilers':'Western',

'Golden Knights':'Western',

'Hurricanes':'Eastern'}

games_team_df['conference'] = games_team_df['team_name'].map(teams_conference_dict)
games_team_df.head(10)
games_regular_season_df = games_team_df.loc[games_team_df['type'] == 'R']

games_playoff_season_df = games_team_df.loc[games_team_df['type'] == 'P']
def computeAccumulatedScore(df):

    df = df.sort_values(by=['team_name', 'date_time_GMT'])

    current_team = ''

    current_score = 0

    score_accumulated_array = [] 

    for i in range(len(df)):

        game_serie = df.iloc[i]

        team = game_serie['team_name']

        if(current_team != team):

            current_team = team

            current_score = 0

        current_score = current_score + game_serie['points']

        score_accumulated_array.append(current_score)



    df['points_accumulated'] = score_accumulated_array

    return df



games_regular_sorted_df = computeAccumulatedScore(games_regular_season_df)

games_regular_sorted_df.head()
games_regular_sorted_df['week_year'] = games_regular_sorted_df['date_time_GMT'].apply(lambda x: x.isocalendar()[1])

first_week = games_regular_sorted_df.iloc[0]['date_time_GMT'].isocalendar()[1]

last_week_year = datetime.datetime(2017, 12, 31, 0, 0).isocalendar()[1]

games_regular_sorted_with_week_df = games_regular_sorted_df.assign(week_season=games_regular_sorted_df['week_year'].apply(lambda x: x-first_week if(x>30) else x+(last_week_year-first_week)))

games_regular_sorted_with_week_df.drop('week_year', inplace=True, axis=1)

games_regular_sorted_with_week_df.head()
#We group by week and team and we get the max num of points accumulated that week by the team. We also sort by the week.

games_regular_division_points_groupby_weekandteam = games_regular_sorted_with_week_df.groupby(['week_season', 'team_name'])

games_regular_division_points_groupby_weekandteam_sorted = games_regular_division_points_groupby_weekandteam.max()[['division','conference','points_accumulated']].reset_index().sort_values(['week_season'])

games_regular_division_points_groupby_weekandteam_sorted.groupby(['conference','division', 'team_name'])[['points_accumulated']].max().sort_values(['conference','division','points_accumulated'],ascending=False)
# We do a for-loop to print the progression of each team in each session

for x in games_regular_division_points_groupby_weekandteam_sorted['division'].unique():

    plt.figure(figsize=(16,8))

    ax = sns.lineplot(x="week_season", y="points_accumulated",

                      hue="team_name", 

                      data=games_regular_division_points_groupby_weekandteam_sorted[games_regular_division_points_groupby_weekandteam_sorted['division'] == x]).set_title("{} division".format(x),fontsize=20)
for x in games_regular_division_points_groupby_weekandteam_sorted['conference'].unique():

    plt.figure(figsize=(16,10))

    sns.lineplot(x="week_season", y="points_accumulated",

                      hue="team_name",

                      data=games_regular_division_points_groupby_weekandteam_sorted[games_regular_division_points_groupby_weekandteam_sorted['conference'] == x]).set_title("{} division".format(x),fontsize=20)

plt.figure(figsize=(16,6))



ax = sns.scatterplot(x='date_time_GMT', y='team_name', s=100,  style="outcome", data=games_playoff_season_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

ax.set(ylabel='Teams', xlabel='Date')

ax.set_title('Play-Offs Run')

ax.plot()
games_playoff_season_df[(games_playoff_season_df['date_time_GMT'] == games_playoff_season_df['date_time_GMT'].max()) & (games_playoff_season_df['outcome'].str.contains('WIN'))]['team_name'].values[0]