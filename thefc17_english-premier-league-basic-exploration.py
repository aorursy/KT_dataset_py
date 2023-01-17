# largely based on https://www.kaggle.com/jonathonv/football-matches-data-analysis, credit to the author for much of this code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("../input/EPL_Set.csv")
len(df)
df.head(5)

df['result'] = 'draw'
df.loc[df['FTHG'] > df['FTAG'], 'result'] = 'home'
df.loc[df['FTAG'] > df['FTHG'], 'result'] = 'visitor'
df.groupby('result')['result'].count()
df.groupby('result')['result'].count().plot(kind='pie', autopct='%1.1f%%', figsize=(4,4))
df['total_goals'] = df['FTAG'] + df['FTHG']
df.groupby('Season')['total_goals'].sum().plot()
# show the number of unique teams that have played in the PL
df['HomeTeam'].nunique()
# show average goals per game per season
ab = df.groupby('Season')['total_goals'].mean().plot(kind="bar", title="Avg. Goals Per Game Per Season", figsize=(12, 8))
ab.set_xlabel("Season")
ab.set_ylabel("Average Goals")
# determine number of games per month and day
df['game_date'] = pd.to_datetime(df['Date'])
df['game_month'] = df['game_date'].dt.month
df['game_weekday'] = df['game_date'].dt.weekday
# by month
df.groupby([df['game_date'].dt.month])["Div"].count().plot(kind='bar')
# by week day - most games are on saturday
df.groupby('game_weekday')['Div'].count().plot(kind='bar')
# where 0 = monday and so forth
# Goals per month
sns.boxplot(x='game_month', y='total_goals', data=df)
# Goals per gameday
sns.boxplot(x='game_weekday', y='total_goals', data=df)
# How many home and visitor wins added as new columns
df = df.merge(pd.get_dummies(df['result']), left_index=True, right_index=True)
df['home_wins_this_season'] = df.groupby(['Season','HomeTeam'])['home'].transform('sum')
df['visitor_wins_this_season'] = df.groupby(['Season','AwayTeam'])['visitor'].transform('sum')
# Which teams win the most home games on average 
(
    df.groupby(['HomeTeam'])['home_wins_this_season']
    .agg(['count','mean'])
    .sort_values(ascending=False, by='mean')
    .round(1)
    .head(10)
)
# Which teams win the most away games on average
(
    df.groupby(['AwayTeam'])['visitor_wins_this_season']
    .agg(['count','mean'])
    .sort_values(ascending=False, by='mean')
    .round(1)
    .head(10)
)
# tally up the results 
visitor_results = (df
                   .groupby(['Season', 'AwayTeam'])['visitor']
                   .sum()
                   .reset_index()
                   .rename(columns={'AwayTeam': 'team',
                                    'visitor': 'visitor_wins'}))

home_results = (df
                 .groupby(['Season', 'HomeTeam'])['home']
                 .sum()
                 .reset_index()
                 .rename(columns={'HomeTeam': 'team',
                                  'home': 'home_wins'}))

wins_per_season = visitor_results.merge(home_results, on=['Season', 'team'])

wins_per_season['total_wins'] = wins_per_season['visitor_wins'] + wins_per_season['home_wins']
wins_per_season.head(5)
# Make a heatmap of wins over time
total_wins_sorted_desc = (wins_per_season
                          .groupby(['team'])['total_wins']
                          .sum()
                          .sort_values(ascending=False)
                          .reset_index()['team'])

wins_per_season_pivot = (wins_per_season
                         .pivot_table(index='team',
                                      columns='Season',
                                      values='total_wins')
                         .fillna(0)
                         .reindex(total_wins_sorted_desc))

plt.figure(figsize=(10, 20))
sns.heatmap(wins_per_season_pivot, cmap='viridis')
# showing dot plot of wins per team per home/away
sns.set(style="whitegrid")
wps = wins_per_season.groupby(['team'])['total_wins','home_wins','visitor_wins'].sum().reset_index()
g = sns.PairGrid(wps.sort_values("total_wins", ascending=False),
                 x_vars=wps.columns[1:], y_vars=["team"],
                 size=10, aspect=.25)

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h",
      palette="Reds_r", edgecolor="gray")

# Use the same x axis limits on all columns and add better labels
g.set(xlabel="Wins", ylabel="")

# Add titles for the columns
titles = ["Total Wins", "Home Wins", "Away Wins"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)

sns.despine(left=True, bottom=True)