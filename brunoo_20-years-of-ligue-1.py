import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as rc



df = pd.read_csv("../input/ligue-1-results-1999-to-2019/Ligue1 Championship.csv")
# Columns for Home team and away team Points

conditions = [(df['Home Team'] == df['Winner']),(df['Home Team'] == df['Loser'])]

values = [3,0]

values1 = [0,3]

df['Home Team Points'] = np.select(conditions, values, default=1)

df['Away Team Points'] = np.select(conditions, values1, default=1)
# Games per season

season = df.groupby('Season')['Season'].count()

season = pd.DataFrame(season)

season.columns = ['Games']

season.reset_index(level=0, inplace=True)



# Goals per season

season1 = df.groupby('Season')['Score'].sum()

season1 = pd.DataFrame(season1)

season1.columns = ['Total Goals']

season1.reset_index(level=0, inplace=True)



# Home goals by season

season2 = df.groupby('Season')['Home Team Goals'].sum()

season2 = pd.DataFrame(season2)

season2.columns = ['Home Goals']

season2.reset_index(level=0, inplace=True)



# Away goals by season

season3 = df.groupby('Season')['Away Team Goals'].sum()

season3 = pd.DataFrame(season3)

season3.columns = ['Away Goals']

season3.reset_index(level=0, inplace=True)



# Home Team Points

season4 = df.groupby('Season')['Home Team Points'].sum()

season4 = pd.DataFrame(season4)

season4.columns = ['Home Team Points']

season4.reset_index(level=0, inplace=True)



# Away Team Points

season5 = df.groupby('Season')['Away Team Points'].sum()

season5 = pd.DataFrame(season5)

season5.columns = ['Away Team Points']

season5.reset_index(level=0, inplace=True)



# Merging dataframes

season = season.merge(season1, how='left', on='Season')

season = season.merge(season2, how='left', on='Season')

season = season.merge(season3, how='left', on='Season')

season = season.merge(season4, how='left', on='Season')

season = season.merge(season5, how='left', on='Season')



# Goals per game

season['Goals per game'] = round(season['Total Goals']/season['Games'],2)
r = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]



names = ('1999/00','2000/01','2001/02','2002/03','2003/04','2004/05','2005/06','2006/07','2007/08','2008/09',

         '2009/10','2010/11','2011/12','2012/13','2013/14','2014/15','2015/16','2016/17','2017/18','2018/19')



column1 = season['Total Goals']



# bar width

barWidth = 0.9



# Goals bar

plt.bar(r, column1, color='darkblue', width=barWidth)



# Axis

plt.xticks(r, names)

plt.xlabel("Seasons")

plt.ylabel("Goals")

plt.title("Goals by Season")

plt.ylim(600, 1100)



# Horizontal gridlines

axes = plt.gca()

axes.yaxis.grid()



# Chart size

plt.rcParams["figure.figsize"] = [24,8]

 

# Show Chart

plt.show()
column1 = season['Goals per game']



# Bar width

barWidth = 0.9



# Goals per Game bar

plt.bar(r, column1, color='darkblue', width=barWidth)



# Axis

plt.xticks(r, names)

plt.xlabel("Seasons")

plt.ylabel("Goals per game")

plt.title("Goals per game by Season")

plt.ylim(2, 2.8)



# Horizontal gridlines

axes = plt.gca()

axes.yaxis.grid()



# Chart size

plt.rcParams["figure.figsize"] = [24,6]

 

# Show chart

plt.show()
column1 = season['Home Goals']/season['Total Goals']

column2 = season['Away Goals']/season['Total Goals']



# Bars width

barWidth = 0.9



# Home Goals bar

plt.bar(r, column1, color='darkblue', edgecolor='white', width=barWidth,label='Home Goals')

# Away Goals bar

plt.bar(r, column2, bottom=column1, color='red', edgecolor='white', width=barWidth,label='Away Goals')



# Axis

plt.xticks(r, names)

plt.xlabel("Seasons")

plt.ylabel("Proportion")

plt.title("Proportion of Home and Away Teams Goals")



# Horizontal gridlines

axes = plt.gca()

axes.yaxis.grid()



# Char Size

plt.rcParams["figure.figsize"] = [24,8]

 

# Show chart

plt.legend()

plt.show()
column1 = season['Home Team Points']/(season['Home Team Points']+season['Away Team Points'])

column2 = season['Away Team Points']/(season['Home Team Points']+season['Away Team Points'])



# Bars width

barWidth = 0.9



# Home Goals bar

plt.bar(r, column1, color='darkblue', edgecolor='white', width=barWidth,label='Home Points')

# Away Goals bar

plt.bar(r, column2, bottom=column1, color='red', edgecolor='white', width=barWidth,label='Away Points')



# Axis

plt.xticks(r, names)

plt.xlabel("Seasons")

plt.ylabel("Proportion")

plt.title("Proportion of Home and Away Teams Points")



# Horizontal gridlines

axes = plt.gca()

axes.yaxis.grid()



# Char Size

plt.rcParams["figure.figsize"] = [24,8]

 

# Show chart

plt.legend()

plt.show()
# Games by team

home_games = df.groupby('Home Team')['Home Team'].count()

home_games = pd.DataFrame(home_games)

home_games.columns = ['Games']

home_games.reset_index(level=0, inplace=True)



# Goals scored by team

home_games1 = df.groupby('Home Team')['Home Team Goals'].sum()

home_games1 = pd.DataFrame(home_games1)

home_games1.columns = ['Goals Scored']

home_games1.reset_index(level=0, inplace=True)



# Goals against by team

home_games2 = df.groupby('Home Team')['Away Team Goals'].sum()

home_games2 = pd.DataFrame(home_games2)

home_games2.columns = ['Goals Against']

home_games2.reset_index(level=0, inplace=True)



# Wins by team

df['Home Team'] = df['Home Team'].astype('category')

home_games3 = df[df['Winner'] == df['Home Team']].groupby(['Home Team']).size().reset_index(name='Wins')



# Loss by team

df['Home Team'] = df['Home Team'].astype('category')

home_games4 = df[df['Winner'] == df['Away Team']].groupby(['Home Team']).size().reset_index(name='Loss')



# Draws by team

df['Home Team'] = df['Home Team'].astype('category')

home_games5 = df[df['Winner'] == 'Draw'].groupby(['Home Team']).size().reset_index(name='Draws')



# Merging dataframes

home_games = home_games.merge(home_games1, how='left', on='Home Team')

home_games = home_games.merge(home_games2, how='left', on='Home Team')

home_games = home_games.merge(home_games3, how='left', on='Home Team')

home_games = home_games.merge(home_games4, how='left', on='Home Team')

home_games = home_games.merge(home_games5, how='left', on='Home Team')



# Goals scored per game

home_games['Goals scored per game'] = round(home_games['Goals Scored']/home_games['Games'],2)



# Goals against per game

home_games['Goals against per game'] = round(home_games['Goals Against']/home_games['Games'],2)



# Create 'Proportion Wins' column

home_games['% Wins'] = 100*round(home_games['Wins']/home_games['Games'],3)



# Create 'Proportion Loss' column

home_games['% Loss'] = 100*round(home_games['Loss']/home_games['Games'],3)



# Create 'Proportion Draws' column

home_games['% Draws'] = 100*round(home_games['Draws']/home_games['Games'],3)



# Create 'Aprov' column

home_games['% Points Performance'] = 100*round((3*home_games['Wins']+home_games['Draws'])/(3*home_games['Games']),3)
# Order dataframe by '% Wins'

home_games.sort_values(by=['% Wins'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='Home Team', y='% Wins',color ='darkblue',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Order dataframe by 'Goals per game'

home_games.sort_values(by=['Goals scored per game'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='Home Team', y='Goals scored per game',color ='darkblue',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Order dataframe by 'Goals Suf per game'

home_games.sort_values(by=['Goals against per game'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='Home Team', y='Goals against per game',color ='darkblue',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
home_games.sort_values(by=['% Points Performance'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='Home Team', y='% Points Performance',color ='darkblue',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Games by team

away_games = df.groupby('Away Team')['Away Team'].count()

away_games = pd.DataFrame(away_games)

away_games.columns = ['Games']

away_games.reset_index(level=0, inplace=True)



# Goals scored by team

away_games1 = df.groupby('Away Team')['Away Team Goals'].sum()

away_games1 = pd.DataFrame(away_games1)

away_games1.columns = ['Goals Scored']

away_games1.reset_index(level=0, inplace=True)



# Goals against by team

away_games2 = df.groupby('Away Team')['Home Team Goals'].sum()

away_games2 = pd.DataFrame(away_games2)

away_games2.columns = ['Goals Against']

away_games2.reset_index(level=0, inplace=True)



# Wins by team

df['Away Team'] = df['Away Team'].astype('category')

away_games3 = df[df['Winner'] == df['Away Team']].groupby(['Away Team']).size().reset_index(name='Wins')



# Loss by team

df['Away Team'] = df['Away Team'].astype('category')

away_games4 = df[df['Winner'] == df['Home Team']].groupby(['Away Team']).size().reset_index(name='Loss')



# Draws by team

df['Away Team'] = df['Away Team'].astype('category')

away_games5 = df[df['Winner'] == 'Draw'].groupby(['Away Team']).size().reset_index(name='Draws')



# Merging dataframes

away_games = away_games.merge(away_games1, how='left', on='Away Team')

away_games = away_games.merge(away_games2, how='left', on='Away Team')

away_games = away_games.merge(away_games3, how='left', on='Away Team')

away_games = away_games.merge(away_games4, how='left', on='Away Team')

away_games = away_games.merge(away_games5, how='left', on='Away Team')



# Goals scored per game

away_games['Goals scored per game'] = round(away_games['Goals Scored']/away_games['Games'],2)



# Goals against per game

away_games['Goals against per game'] = round(away_games['Goals Against']/away_games['Games'],2)



# Create 'Proportion Wins' column

away_games['% Wins'] = 100*round(away_games['Wins']/away_games['Games'],3)



# Create 'Proportion Loss' column

away_games['% Loss'] = 100*round(away_games['Loss']/away_games['Games'],3)



# Create 'Proportion Draws' column

away_games['% Draws'] = 100*round(away_games['Draws']/away_games['Games'],3)



# Create 'Aprov' column

away_games['% Points Performance'] = 100*round((3*away_games['Wins']+away_games['Draws'])/(3*away_games['Games']),3)
# Order dataframe by '% Wins'

away_games.sort_values(by=['% Wins'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='Away Team', y='% Wins',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Order dataframe by 'Goals per game'

away_games.sort_values(by=['Goals scored per game'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='Away Team', y='Goals scored per game',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Order dataframe by 'Goals against per game'

away_games.sort_values(by=['Goals against per game'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='Away Team', y='Goals against per game',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
away_games.sort_values(by=['% Points Performance'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='Away Team', y='% Points Performance',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Make some changes

home_results = home_games.copy()

home_results = home_results.drop(['Goals Scored', 'Goals Against','Goals scored per game', 'Goals against per game',

                                  '% Wins','% Loss','% Draws','% Points Performance'], axis = 1)

home_results = home_results.rename(columns={"Home Team":"Team","Games":"Home Games","Wins":"Wins Home","Loss":"Loss Home",

                               "Draws":"Draws Home"})



away_results = away_games.copy()

away_results = away_results.drop(['Goals Scored', 'Goals Against','Goals scored per game', 'Goals against per game',

                                  '% Wins', '% Loss','% Draws','% Points Performance'], axis = 1)

away_results = away_results.rename(columns={"Away Team":"Team","Games":"Away Games","Wins":"Wins Away","Loss":"Loss Away",

                                                "Draws":"Draws Away"})



# Merger dataframes

games_results = home_results.merge(away_results, how='left', on='Team')



# Create column 'Total Games'

games_results['Total Games'] = games_results['Home Games'] + games_results['Away Games']



# Create column 'Total Wins'

games_results['Total Wins'] = games_results['Wins Home'] + games_results['Wins Away']



# Create column 'Total Loss'

games_results['Total Loss'] = games_results['Loss Home'] + games_results['Loss Away']



# Create column 'Total Draws'

games_results['Total Draws'] = games_results['Draws Home'] + games_results['Draws Away']



# Create column 'Home points'

games_results['Home points'] = 3*games_results['Wins Home'] + games_results['Draws Home']



# Create column 'Away points'

games_results['Away points'] = 3*games_results['Wins Away'] + games_results['Draws Away']



# Create column 'Total points'

games_results['Total points'] = games_results['Home points'] + games_results['Away points']



# Create column 'Points performance'

games_results['Points Performance'] = 100*round((3*games_results['Total Wins']+games_results['Total Draws'])/

                                                (3*games_results['Total Games']),3)



# Create column '% Wins at home'

games_results['% Wins Home'] = 100*round(games_results['Wins Home']/games_results['Total Wins'],3)



# Create column '% Wins Away'

games_results['% Wins Away'] = 100*round(games_results['Wins Away']/games_results['Total Wins'],3)



# Create column '% Home points'

games_results['% Home points'] = 100*round(games_results['Home points']/games_results['Total points'],3)



# Create column '% Away points'

games_results['% Away points'] = 100*round(games_results['Away points']/games_results['Total points'],3)
games_results.sort_values(by=['Total Wins'], inplace=True, ascending=False)

ax = games_results.plot.barh(x='Team', y='Total Wins',color ='darkblue',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Sort dataframe

games_results.sort_values(by=['% Wins Home'], inplace=True, ascending=False)



# Create columns for teams

team = games_results['Team'].tolist()



# Create columns for wins at home and wins away

win_home = games_results['% Wins Home'].to_numpy()

win_away = games_results['% Wins Away'].to_numpy()

values = np.vstack((win_home, win_away)).T



# Create new dataframe

prop_win = pd.DataFrame(values, team)



# Define color

color = ['darkblue','red']



# Define legend

labels = ['% Wins Home','% Wins Away']



# Plot chart

prop_win.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
games_results.sort_values(by=['Total points'], inplace=True, ascending=False)

ax = games_results.plot.barh(x='Team', y='Total points',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_results.sort_values(by=['Points Performance'], inplace=True, ascending=False)

ax = games_results.plot.barh(x='Team', y='Points Performance',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Sort dataframe

games_results.sort_values(by=['% Home points'], inplace=True, ascending=False)



# Create columns for teams

team = games_results['Team'].tolist()



# Create columns for wins at home and wins away

points_home = games_results['% Home points'].to_numpy()

points_away = games_results['% Away points'].to_numpy()

values = np.vstack((points_home, points_away)).T



# Create new dataframe

prop_win = pd.DataFrame(values, team)



# Define color

color = ['darkblue','red']



# Define legend

labels = ['% Home points','% Away points']



# Plot chart

prop_win.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
# Make some changes

home_goals = home_games.copy()

home_goals = home_goals.drop(['Wins','Loss','Draws','Goals scored per game','Goals against per game','% Wins',

                             '% Loss','% Draws','% Points Performance'], axis = 1)

home_goals = home_goals.rename(columns={"Home Team":"Team","Games":"Home Games","Goals Scored":"Goals Scored Home",

                                       "Goals Against":"Goals Against Home"})



away_goals = away_games.copy()

away_goals = away_goals.drop(['Wins','Loss','Draws','Goals scored per game','Goals against per game','% Wins',

                             '% Loss','% Draws','% Points Performance'], axis = 1)

away_goals = away_goals.rename(columns={"Away Team":"Team","Games":"Away Games","Goals Scored":"Goals Scored Away",

                                       "Goals Against":"Goals Against Away"})



# Merger datafrmaes

games_goals = home_goals.merge(away_goals, how='left', on='Team')



# Create 'Total Games' column

games_goals['Total Games'] = games_goals['Home Games'] + games_goals['Away Games']



# Create 'Total goals scored' column

games_goals['Total goals scored'] = games_goals['Goals Scored Home'] + games_goals['Goals Scored Away']



# Create 'Total goals against' column

games_goals['Total goals against'] = games_goals['Goals Against Home'] + games_goals['Goals Against Away']



# Create 'Total goals scored per game' column

games_goals['Total goals scored per game'] = round(games_goals['Total goals scored']/games_goals['Total Games'],2)



# Create 'Total goals against per game' column

games_goals['Total goals against per game'] = round(games_goals['Total goals against']/games_goals['Total Games'],2)



# Create '% goals scored home' column

games_goals['% goals scored home'] = 100*round(games_goals['Goals Scored Home']/games_goals['Total goals scored'],3)



# Create '% goals scored away' column

games_goals['% goals scored away'] = 100*round(games_goals['Goals Scored Away']/games_goals['Total goals scored'],3)



# Create '% goals against home' column

games_goals['% goals against home'] = 100*round(games_goals['Goals Against Home']/games_goals['Total goals against'],3)



# Create '% goals against away' column

games_goals['% goals against away'] = 100*round(games_goals['Goals Against Away']/games_goals['Total goals against'],3)



# Create 'Dif goals' column -- Difference between total goals scored and total goals against

games_goals['Dif Goals'] = games_goals['Total goals scored'] - games_goals['Total goals against']
games_goals.sort_values(by=['Total goals scored'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Total goals scored',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_goals.sort_values(by=['Total goals scored per game'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Total goals scored per game',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_goals.sort_values(by=['Total goals against'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Total goals against',color ='red',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_goals.sort_values(by=['Dif Goals'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Dif Goals',color ='darkblue',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Sort dataframe

games_goals.sort_values(by=['% goals scored home'], inplace=True, ascending=False)



# Create columns for teams

team = games_goals['Team'].tolist()



# Create columns for goals at home and goals away

goals_home = games_goals['% goals scored home'].to_numpy()

goals_away = games_goals['% goals scored away'].to_numpy()

values = np.vstack((goals_home, goals_away)).T



# Create new dataframe

prop_win = pd.DataFrame(values, team)



# Define color

color = ['darkblue','red']



# Define legend

labels = ['% Home goals','% Away goals']



# Plot chart

prop_win.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
# Sort dataframe

games_goals.sort_values(by=['% goals against home'], inplace=True, ascending=False)



# Create columns for teams

team = games_goals['Team'].tolist()



# Create columns for goals against at home and goals away

goals_a_home = games_goals['% goals against home'].to_numpy()

goals_a_away = games_goals['% goals against away'].to_numpy()

values = np.vstack((goals_a_home, goals_a_away)).T



# Create new dataframe

prop_win = pd.DataFrame(values, team)



# Define color

color = ['darkblue','red']



# Define legend

labels = ['% Home goals against','% Away goals against']



# Plot chart

prop_win.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
goals = games_goals['Total goals scored'].sum()

print("Goals scored: "+str(goals))



games = games_goals['Total Games'].sum()

goals_per_game = round(goals/games,2)

print("Goals scored per game: "+str(goals_per_game))



b_winner = games_results.nlargest(1, 'Total Wins')

winner = b_winner.iloc[0][0]

wins = b_winner.iloc[0][10]

print("Most wins: "+str(winner)+ " with " + str(wins)+ " wins.")



b_goals = games_goals.nlargest(1, 'Total goals scored')

team = b_goals.iloc[0][0]

goals = b_goals.iloc[0][8]

print("Most goals: "+str(team)+ " with " + str(goals)+ " goals.")





b_goals_per_game = games_goals.nlargest(1, 'Total goals scored per game')

team = b_goals_per_game.iloc[0][0]

gpg = b_goals_per_game.iloc[0][10]

print("Most goals per game: "+str(team)+ " with " + str(gpg)+ " goals.")



b_points = games_results.nlargest(1, 'Total points')

team = b_points.iloc[0][0]

points = b_points.iloc[0][15]

print("Most points earned: "+str(team)+ " with " + str(points)+ " points.")