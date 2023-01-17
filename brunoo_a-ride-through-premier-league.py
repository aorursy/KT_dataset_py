# Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as rc



# Import data

df = pd.read_csv("../input/english-premier-league-results/EPL.csv")

df.head()
df.shape
# Let's drop columns we won't use

df = df.drop(['FTR','HTR','Referee', 'HS', 'AS', 'HST', 'AST', 'HC' ,'AC', 'HF' ,'AF', 'HY', 'AY', 'HR' ,'AR'], axis = 1)



# Let's modify data type in 'Date' column



df['month'] = [x.strip()[3:5] for x in df['Date']]

df['year'] = [x.strip()[-2:] for x in df['Date']]

df['month'] = pd.to_numeric(df['month'])

df['year'] = pd.to_numeric(df['year'])





# Let's create a column for 'Score'

df['Score'] = df['FTHG'] + df['FTAG']



# Let's create columns 'Final Winner' and 'Final Loser'

conditions = [(df['FTHG'] > df['FTAG']),(df['FTHG'] < df['FTAG'])]

values = [df['HomeTeam'],df['AwayTeam']]

values1 = [df['AwayTeam'],df['HomeTeam']]

df['Final Winner'] = np.select(conditions, values, default='Draw')

df['Final Loser'] = np.select(conditions, values1, default='Draw')



# Let's create columns 'Halftime Winner' and 'Halftime Loser'

conditions1 = [(df['HTHG'] > df['HTAG']),(df['HTHG'] < df['HTAG'])]

values2 = [df['HomeTeam'],df['AwayTeam']]

values3 = [df['AwayTeam'],df['HomeTeam']]

df['Halftime Winner'] = np.select(conditions1, values2, default='Draw')

df['Halftime Loser'] = np.select(conditions1, values3, default='Draw')



# Let's create column for comebacks

df['Comeback'] = np.where((df['Final Winner'] != df['Halftime Winner'])&(df['Final Winner']!='Draw')&(df['Halftime Winner']!='Draw'),

                          1, 0)



# Let's create a column for season

conditions2 = [(df['year'] == 0)&(df['month'] > 7),(df['year'] == 1)&(df['month'] < 7),

             (df['year'] == 1)&(df['month'] > 7),(df['year'] == 2)&(df['month'] < 7),

             (df['year'] == 2)&(df['month'] > 7),(df['year'] == 3)&(df['month'] < 7),

             (df['year'] == 3)&(df['month'] > 7),(df['year'] == 4)&(df['month'] < 7),

             (df['year'] == 4)&(df['month'] > 7),(df['year'] == 5)&(df['month'] < 7),

             (df['year'] == 5)&(df['month'] > 7),(df['year'] == 6)&(df['month'] < 7),

             (df['year'] == 6)&(df['month'] > 7),(df['year'] == 7)&(df['month'] < 7), 

             (df['year'] == 7)&(df['month'] > 7),(df['year'] == 8)&(df['month'] < 7),

             (df['year'] == 8)&(df['month'] > 7),(df['year'] == 9)&(df['month'] < 7),

             (df['year'] == 9)&(df['month'] > 7),(df['year'] == 10)&(df['month'] < 7),

             (df['year'] == 10)&(df['month'] > 7),(df['year'] == 11)&(df['month'] < 7),

             (df['year'] == 11)&(df['month'] > 7),(df['year'] == 12)&(df['month'] < 7),

             (df['year'] == 12)&(df['month'] > 7),(df['year'] == 13)&(df['month'] < 7),

             (df['year'] == 13)&(df['month'] > 7),(df['year'] == 14)&(df['month'] < 7),  

             (df['year'] == 14)&(df['month'] > 7),(df['year'] == 15)&(df['month'] < 7),

             (df['year'] == 15)&(df['month'] > 7),(df['year'] == 16)&(df['month'] < 7),

             (df['year'] == 16)&(df['month'] > 7),(df['year'] == 17)&(df['month'] < 7),

             (df['year'] == 17)&(df['month'] > 7),(df['year'] == 18)&(df['month'] < 7),

             (df['year'] == 18)&(df['month'] > 7),(df['year'] == 19)&(df['month'] < 7),

             (df['year'] == 19)&(df['month'] > 7),(df['year'] == 20)&(df['month'] < 7)]



values4 = ['2000/01','2000/01','2001/02','2001/02','2002/03','2002/03','2003/04','2003/04','2004/05','2004/05','2005/06',

           '2005/06','2006/07','2006/07','2007/08','2007/08','2008/09','2008/09','2009/10','2009/10','2010/11','2010/11',

           '2011/12','2011/12','2012/13','2012/13','2013/14','2013/14','2014/15','2014/15','2015/16','2015/16','2016/17',

           '2016/17','2017/18','2017/18','2018/19','2018/19','2019/20','2019/20']



df['Season'] = np.select(conditions2, values4,default='X')



# Let's drop columns

df = df.drop(['month','year'], axis = 1)



# Let's drop every game from 2019/20 season

df = df.drop(df[(df['Season'] == '2019/20')].index)

df = df.drop(df[(df['Season'] == 'X')].index)
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

season2 = df.groupby('Season')['FTHG'].sum()

season2 = pd.DataFrame(season2)

season2.columns = ['Home Goals']

season2.reset_index(level=0, inplace=True)



# Away goals by season

season3 = df.groupby('Season')['FTAG'].sum()

season3 = pd.DataFrame(season3)

season3.columns = ['Away Goals']

season3.reset_index(level=0, inplace=True)



# Comebacks per season

season4 = df.groupby('Season')['Comeback'].sum()

season4 = pd.DataFrame(season4)

season4.columns = ['Comeback']

season4.reset_index(level=0, inplace=True)



# Merging dataframes

season = season.merge(season1, how='left', on='Season')

season = season.merge(season2, how='left', on='Season')

season = season.merge(season3, how='left', on='Season')

season = season.merge(season4, how='left', on='Season')



# Goals per game

season['Goals per game'] = round(season['Total Goals']/season['Games'],2)
print(season)
r = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]



names = ('2000/01','2001/02','2002/03','2003/04','2004/05','2005/06','2006/07','2007/08','2008/09',

         '2009/10','2010/11','2011/12','2012/13','2013/14','2014/15','2015/16','2016/17','2017/18','2018/19')



column1 = season['Total Goals']



# bar width

barWidth = 0.9



# Goals bar

plt.bar(r, column1, color='indigo', width=barWidth)



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



# bar width

barWidth = 0.9



# Goals bar

plt.bar(r, column1, color='indigo', width=barWidth)



# Axis

plt.xticks(r, names)

plt.xlabel("Seasons")

plt.ylabel("Goals")

plt.title("Goals per game")

plt.ylim(2.4, 2.9)



# Horizontal gridlines

axes = plt.gca()

axes.yaxis.grid()



# Chart size

plt.rcParams["figure.figsize"] = [24,8]

 

# Show Chart

plt.show()
column1 = season['Home Goals']/season['Total Goals']

column2 = season['Away Goals']/season['Total Goals']



# Bars width

barWidth = 0.9



# Home Goals bar

plt.bar(r, column1, color='indigo', edgecolor='white', width=barWidth,label='Home Goals')

# Away Goals bar

plt.bar(r, column2, bottom=column1, color='darkorange', edgecolor='white', width=barWidth,label='Away Goals')



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
column1 = season['Comeback']



# bar width

barWidth = 0.9



# Goals bar

plt.bar(r, column1, color='indigo', width=barWidth)



# Axis

plt.xticks(r, names)

plt.xlabel("Seasons")

plt.ylabel("Games")

plt.title("Comebacks")

plt.ylim(5, 25)



# Horizontal gridlines

axes = plt.gca()

axes.yaxis.grid()



# Chart size

plt.rcParams["figure.figsize"] = [24,8]

 

# Show Chart

plt.show()
### AT HOME

home = df.copy()



# Games by team

home_games = df.groupby('HomeTeam')['HomeTeam'].count()

home_games = pd.DataFrame(home_games)

home_games.columns = ['Games']

home_games.reset_index(level=0, inplace=True)



# Goals scored by team

home_games1 = df.groupby('HomeTeam')['FTHG'].sum()

home_games1 = pd.DataFrame(home_games1)

home_games1.columns = ['Goals Scored']

home_games1.reset_index(level=0, inplace=True)



# Goals against by team

home_games2 = df.groupby('HomeTeam')['FTAG'].sum()

home_games2 = pd.DataFrame(home_games2)

home_games2.columns = ['Goals Against']

home_games2.reset_index(level=0, inplace=True)



# Comebacks by team

home['trailer'] = np.where((home['HomeTeam'] == home['Final Winner'])&(home['Comeback']== 1),1, 0)

home_games3 = home.groupby('HomeTeam')['trailer'].sum()

home_games3 = pd.DataFrame(home_games3)

home_games3.columns = ['Comeback']

home_games3.reset_index(level=0, inplace=True)



# Wins by team

home['HomeTeam'] = home['HomeTeam'].astype('category')

home_games4 = home[home['Final Winner'] == home['HomeTeam']].groupby(['HomeTeam']).size().reset_index(name='Wins')



# Loss by team

home_games5 = home[home['Final Winner'] == home['AwayTeam']].groupby(['HomeTeam']).size().reset_index(name='Loss')



# Draws by team

home_games6 = home[home['Final Winner'] == 'Draw'].groupby(['HomeTeam']).size().reset_index(name='Draws')



# Merging dataframes

home_games = home_games.merge(home_games1, how='left', on='HomeTeam')

home_games = home_games.merge(home_games2, how='left', on='HomeTeam')

home_games = home_games.merge(home_games3, how='left', on='HomeTeam')

home_games = home_games.merge(home_games4, how='left', on='HomeTeam')

home_games = home_games.merge(home_games5, how='left', on='HomeTeam')

home_games = home_games.merge(home_games6, how='left', on='HomeTeam')



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



home_games.head()
home_games.sort_values(by=['Goals Scored'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='HomeTeam', y='Goals Scored',color ='darkorange',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
home_games.sort_values(by=['Goals scored per game'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='HomeTeam', y='Goals scored per game',color ='darkorange',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
home_games.sort_values(by=['% Wins'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='HomeTeam', y='% Wins',color ='darkorange',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
home_games.sort_values(by=['% Points Performance'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='HomeTeam', y='% Points Performance',color ='darkorange',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
home_games.sort_values(by=['Comeback'], inplace=True, ascending=False)

ax = home_games.plot.barh(x='HomeTeam', y='Comeback',color ='darkorange',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
### AWAY

away = df.copy()



# Games by team

away_games = df.groupby('AwayTeam')['AwayTeam'].count()

away_games = pd.DataFrame(away_games)

away_games.columns = ['Games']

away_games.reset_index(level=0, inplace=True)



# Goals scored by team

away_games1 = df.groupby('AwayTeam')['FTAG'].sum()

away_games1 = pd.DataFrame(away_games1)

away_games1.columns = ['Goals Scored']

away_games1.reset_index(level=0, inplace=True)



# Goals against by team

away_games2 = df.groupby('AwayTeam')['FTHG'].sum()

away_games2 = pd.DataFrame(away_games2)

away_games2.columns = ['Goals Against']

away_games2.reset_index(level=0, inplace=True)



# Comebacks by team

away['trailer'] = np.where((away['AwayTeam'] == away['Final Winner'])&(away['Comeback']== 1),1, 0)

away_games3 = away.groupby('AwayTeam')['trailer'].sum()

away_games3 = pd.DataFrame(away_games3)

away_games3.columns = ['Comeback']

away_games3.reset_index(level=0, inplace=True)



# Wins by team

away['AwayTeam'] = away['AwayTeam'].astype('category')

away_games4 = away[away['Final Winner'] == away['AwayTeam']].groupby(['AwayTeam']).size().reset_index(name='Wins')



# Loss by team

away_games5 = away[away['Final Winner'] == away['HomeTeam']].groupby(['AwayTeam']).size().reset_index(name='Loss')



# Draws by team

away_games6 = away[away['Final Winner'] == 'Draw'].groupby(['AwayTeam']).size().reset_index(name='Draws')



# Merging dataframes

away_games = away_games.merge(away_games1, how='left', on='AwayTeam')

away_games = away_games.merge(away_games2, how='left', on='AwayTeam')

away_games = away_games.merge(away_games3, how='left', on='AwayTeam')

away_games = away_games.merge(away_games4, how='left', on='AwayTeam')

away_games = away_games.merge(away_games5, how='left', on='AwayTeam')

away_games = away_games.merge(away_games6, how='left', on='AwayTeam')



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



away_games.head()
away_games.sort_values(by=['Goals Scored'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='AwayTeam', y='Goals Scored',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
away_games.sort_values(by=['Goals scored per game'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='AwayTeam', y='Goals scored per game',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
away_games.sort_values(by=['% Wins'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='AwayTeam', y='% Wins',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
away_games.sort_values(by=['% Points Performance'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='AwayTeam', y='% Points Performance',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
away_games.sort_values(by=['Comeback'], inplace=True, ascending=False)

ax = away_games.plot.barh(x='AwayTeam', y='Comeback',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Adjust for home games

home_results = home_games.copy()

home_results = home_results.drop(['Goals Scored','Goals Against','Goals scored per game','Goals against per game',

                                  '% Wins','% Loss','% Draws','% Points Performance'], axis = 1)

home_results.columns = ['Team','Home Games','Home Comebacks','Home Wins','Home Losses','Home Draws']



# Adjusts for away games

away_results = away_games.copy()

away_results = away_results.drop(['Goals Scored','Goals Against','Goals scored per game','Goals against per game',

                                  '% Wins','% Loss','% Draws','% Points Performance'], axis = 1)



away_results.columns = ['Team','Away Games','Away Comebacks','Away Wins','Away Losses','Away Draws']



# Merge dataframes

games_results = home_results.merge(away_results, how='left', on='Team')



#Create Columns

games_results['Games'] = games_results['Home Games'] + games_results['Away Games']

games_results['Comebacks'] = games_results['Home Comebacks'] + games_results['Away Comebacks']

games_results['Wins'] = games_results['Home Wins'] + games_results['Away Wins']

games_results['Losses'] = games_results['Home Losses'] + games_results['Away Losses']

games_results['Draws'] = games_results['Home Draws'] + games_results['Away Draws']

games_results['% Wins'] = 100*round(games_results['Wins']/games_results['Games'],3)

games_results['% Wins home'] = 100*round(games_results['Home Wins']/games_results['Wins'],3)

games_results['% Wins away'] = 100*round(games_results['Away Wins']/games_results['Wins'],3)

games_results['Points Performance'] = 100*round((3*games_results['Wins']+games_results['Draws'])/

                                                (3*games_results['Games']),3)

games_results['% Points home'] = 100*round((3*games_results['Home Wins']+games_results['Home Draws'])/

                                           (3*games_results['Wins']+games_results['Draws']),3)

games_results['% Points away'] = 100*round((3*games_results['Away Wins']+games_results['Away Draws'])/

                                           (3*games_results['Wins']+games_results['Draws']),3)

games_results['% home comebacks'] = 100*round(games_results['Home Comebacks']/games_results['Comebacks'],3)

games_results['% away comebacks'] = 100*round(games_results['Away Comebacks']/games_results['Comebacks'],3)



# Drop some columns

games_results = games_results.drop(['Home Games','Home Comebacks','Home Wins','Home Losses','Home Draws',

                                 'Away Games','Away Comebacks','Away Wins','Away Losses','Away Draws'], axis = 1)



games_results.head()
# Sort dataframe

games_results.sort_values(by=['% Wins home'], inplace=True, ascending=False)



# Create column for team

team = games_results['Team'].tolist()



# Create columns for wins at home and wins away

win_home = games_results['% Wins home'].to_numpy()

win_away = games_results['% Wins away'].to_numpy()

values = np.vstack((win_home, win_away)).T



# Create new dataframe

prop_win = pd.DataFrame(values, team)



# Define color

color = ['darkorange','indigo']



# Define legend

labels = ['% Wins Home','% Wins Away']



# Plot chart

prop_win.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
# Sort dataframe

games_results.sort_values(by=['% Points home'], inplace=True, ascending=False)



# Create column for team

team = games_results['Team'].tolist()



# Create columns for wins at home and wins away

point_home = games_results['% Points home'].to_numpy()

point_away = games_results['% Points away'].to_numpy()

values = np.vstack((point_home, point_away)).T



# Create new dataframe

prop_point = pd.DataFrame(values, team)



# Define color

color = ['darkorange','indigo']



# Define legend

labels = ['% Points Home','% Points Away']



# Plot chart

prop_point.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
games_results.sort_values(by=['% Wins'], inplace=True, ascending=False)

ax = games_results.plot.barh(x='Team', y='% Wins',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_results.sort_values(by=['Points Performance'], inplace=True, ascending=False)

ax = games_results.plot.barh(x='Team', y='Points Performance',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_results.sort_values(by=['Comebacks'], inplace=True, ascending=False)

ax = games_results.plot.barh(x='Team', y='Comebacks',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Sort dataframe

games_results.sort_values(by=['% home comebacks'], inplace=True, ascending=False)



# Create column for team

team = games_results['Team'].tolist()



# Create columns for wins at home and wins away

comeback_home = games_results['% home comebacks'].to_numpy()

comeback_away = games_results['% away comebacks'].to_numpy()

values = np.vstack((comeback_home, comeback_away)).T



# Create new dataframe

prop_comeback = pd.DataFrame(values, team)



# Define color

color = ['darkorange','indigo']



# Define legend

labels = ['% home comebacks','% away comebacks']



# Plot chart

prop_comeback.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
# Adjust for home games

home_goals = home_games.copy()

home_goals = home_goals.drop(['Comeback','Wins','Loss','Draws','Goals scored per game','Goals against per game',

                              '% Wins','% Loss','% Draws','% Points Performance'], axis = 1)

home_goals.columns = ['Team','Home games','Home Goals Scored','Home Goals Against']



# Adjust for away games

away_goals = away_games.copy()

away_goals = away_goals.drop(['Comeback','Wins','Loss','Draws','Goals scored per game','Goals against per game',

                              '% Wins','% Loss','% Draws','% Points Performance'], axis = 1)

away_goals.columns = ['Team','Away games','Away Goals Scored','Away Goals Against']



# Merge dataframes

games_goals = home_goals.merge(away_goals, how='left', on='Team')



# Create Columns

games_goals['Games'] = games_goals['Home games'] + games_goals['Away games']

games_goals['Goals scored'] = games_goals['Home Goals Scored'] + games_goals['Away Goals Scored']

games_goals['Goals scored per game'] = round(games_goals['Goals scored']/games_goals['Games'],3)

games_goals['Goals against'] = games_goals['Home Goals Against'] + games_goals['Away Goals Against']

games_goals['Goals against per game'] = round(games_goals['Goals against']/games_goals['Games'],3)

games_goals['% Goals scored home'] = round(games_goals['Home Goals Scored']/games_goals['Goals scored'],3)

games_goals['% Goals scored away'] = round(games_goals['Away Goals Scored']/games_goals['Goals scored'],3)

games_goals['% Goals against home'] = round(games_goals['Home Goals Against']/games_goals['Goals against'],3)

games_goals['% Goals against away'] = round(games_goals['Away Goals Against']/games_goals['Goals against'],3)



# Drop some columns

games_goals = games_goals.drop(['Home games','Home Goals Scored','Home Goals Against','Away games','Away Goals Scored',

                                 'Away Goals Against'], axis = 1)



games_goals.head()
games_goals.sort_values(by=['Goals scored'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Goals scored',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_goals.sort_values(by=['Goals scored per game'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Goals scored per game',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Sort dataframe

games_goals.sort_values(by=['% Goals scored home'], inplace=True, ascending=False)



# Create column for team

team = games_goals['Team'].tolist()



# Create columns for wins at home and wins away

goal_home = games_goals['% Goals scored home'].to_numpy()

goal_away = games_goals['% Goals scored away'].to_numpy()

values = np.vstack((goal_home, goal_away)).T



# Create new dataframe

prop_goal = pd.DataFrame(values, team)



# Define color

color = ['darkorange','indigo']



# Define legend

labels = ['% Goals scored home','% Goals scored away']



# Plot chart

prop_point.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
games_goals.sort_values(by=['Goals against'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Goals against',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
games_goals.sort_values(by=['Goals against per game'], inplace=True, ascending=False)

ax = games_goals.plot.barh(x='Team', y='Goals against per game',color ='indigo',figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()
# Sort dataframe

games_goals.sort_values(by=['% Goals against home'], inplace=True, ascending=False)



# Create column for team

team = games_goals['Team'].tolist()



# Create columns for wins at home and wins away

goal_home = games_goals['% Goals against home'].to_numpy()

goal_away = games_goals['% Goals against away'].to_numpy()

values = np.vstack((goal_home, goal_away)).T



# Create new dataframe

prop_goal = pd.DataFrame(values, team)



# Define color

color = ['darkorange','indigo']



# Define legend

labels = ['% Goals against home','% Goals against away']



# Plot chart

prop_point.plot.barh(color = color,stacked=True,figsize=(10,10))

axes = plt.gca()

axes.xaxis.grid()

plt.legend(labels,loc=1)
# Goals scored

goals = games_goals['Goals scored'].sum()

print("Goals scored in the last 19 seasons: "+str(goals))



# Goals per game

games = games_goals['Games'].sum()

goals_per_game = round(goals/games,2)

print("Goals scored per game: "+str(goals_per_game))



# Team with most wins

t_winner = games_results.nlargest(1, 'Wins')

winner = t_winner.iloc[0][0]

wins = t_winner.iloc[0][3]

print("Most wins: "+str(winner)+ " with " + str(wins)+ " wins.")



# Team with most goals

t_goals = games_goals.nlargest(1, 'Goals scored')

team = t_goals.iloc[0][0]

goals = t_goals.iloc[0][2]

print("Most goals: "+str(team)+ " with " + str(goals)+ " goals.")



# Team with most goals per game

t_goals_per_game = games_goals.nlargest(1, 'Goals scored per game')

team = t_goals_per_game.iloc[0][0]

gpg = t_goals_per_game.iloc[0][3]

print("Most goals per game: "+str(team)+ " with " + str(round(gpg,2))+ " goals.")



# Team with best points performance

t_points = games_results.nlargest(1, 'Points Performance')

team = t_points.iloc[0][0]

points = t_points.iloc[0][9]

print("Best point performance: "+str(team)+ " with " + str(round(points,2))+ "%.")