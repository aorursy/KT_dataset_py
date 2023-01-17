import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# load data

df = pd.read_csv('../input/FMEL_Dataset.csv')

len(df)
# renaming columns to be more 'pythonic' ;)

df.columns = ['id','season','division','round','local_team','visitor_team',

              'local_goals','visitor_goals','date','timestamp']

df = df.drop(['id','timestamp'], axis=1)
# inspect the data

df.head(5)
# Creating a result column

df['result'] = 'draw'

df.ix[df['local_goals'] > df['visitor_goals'], 'result'] = 'local'

df.ix[df['visitor_goals'] > df['local_goals'], 'result'] = 'visitor'
# quick look at all results

df.groupby('result')['result'].count()
# Mandatory ugly pie chart, outcome split

df.groupby('result')['result'].count().plot(kind='pie', autopct='%1.1f%%', figsize=(4,4))
df['total_goals'] = df['visitor_goals'] + df['local_goals']
# Number of total goals per season has increased over time

df.groupby('season')['total_goals'].sum().plot()
# Only 6 more teams since 1970

df.groupby('season')['local_team'].nunique().plot()
# Maybe the increase in games explains the increase in goals

df.groupby('season')['round'].count().plot()
# The average number of total goals per season is slowly but steadily increasing

avg_goals_per_season = df.groupby('season')['total_goals'].mean().reset_index()

avg_goals_per_season['season'] = avg_goals_per_season['season'].map(lambda s: int(s[:4]))

sns.regplot(x='season', y='total_goals', data=avg_goals_per_season)
df['game_date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

df['game_month'] = df['game_date'].dt.month

df['game_weekday'] = df['game_date'].dt.weekday
# Number of games per month

df.groupby([df['game_date'].dt.month])['round'].count().plot(kind='bar')
# Higher scoring games leading up to end of season

sns.boxplot(x='game_month', y='total_goals', data=df)
# Score more goals on Mondays

sns.boxplot(x='game_weekday', y='total_goals', data=df)
# Looks like most games are played on the weekend :)

df.groupby('game_weekday')['round'].count().plot(kind='bar')
# Convert result column to three binary columns so they can be summed up later

df = df.merge(pd.get_dummies(df['result']), left_index=True, right_index=True)
# How many local and visitor wins per season added as new columns

df['local_wins_this_season'] = df.groupby(['season','local_team'])['local'].transform('sum')

df['visitor_wins_this_season'] = df.groupby(['season','visitor_team'])['visitor'].transform('sum')
# Which teams win the most local games on average per season

(

    df.groupby(['local_team'])['local_wins_this_season']

    .agg(['count','mean'])

    .sort_values(ascending=False, by='mean')

    .round(1)

    .head(10)

)
# Which teams win the most visitor games on average per season

(

    df.groupby(['visitor_team'])['visitor_wins_this_season']

    .agg(['count','mean'])

    .sort_values(ascending=False, by='mean')

    .round(1)

    .head(10)

)
visitor_results = (df

                   .groupby(['season', 'visitor_team'])['visitor']

                   .sum()

                   .reset_index()

                   .rename(columns={'visitor_team': 'team',

                                    'visitor': 'visitor_wins'}))



local_results = (df

                 .groupby(['season', 'local_team'])['local']

                 .sum()

                 .reset_index()

                 .rename(columns={'local_team': 'team',

                                  'local': 'local_wins'}))



wins_per_season = visitor_results.merge(local_results, on=['season', 'team'])



wins_per_season['total_wins'] = wins_per_season['visitor_wins'] + wins_per_season['local_wins']



wins_per_season.head(5)
total_wins_sorted_desc = (wins_per_season

                          .groupby(['team'])['total_wins']

                          .sum()

                          .sort_values(ascending=False)

                          .reset_index()['team'])



wins_per_season_pivot = (wins_per_season

                         .pivot_table(index='team',

                                      columns='season',

                                      values='total_wins')

                         .fillna(0)

                         .reindex(total_wins_sorted_desc))
# Look at total wins per season over time

plt.figure(figsize=(10, 20))

sns.heatmap(wins_per_season_pivot, cmap='viridis')
# Which wins contributed to the total wins

sns.set(style="whitegrid")



# Rearrange data

top_50 = wins_per_season[wins_per_season['team'].isin(total_wins_sorted_desc[:50])]

wps = top_50[['team','total_wins','local_wins','visitor_wins']]

wps = wps.groupby(['team'])['total_wins','local_wins','visitor_wins'].sum().reset_index()



# Make the PairGrid

g = sns.PairGrid(wps.sort_values("total_wins", ascending=False),

                 x_vars=wps.columns[1:], y_vars=["team"],

                 size=10, aspect=.25)



# Draw a dot plot using the stripplot function

g.map(sns.stripplot, size=10, orient="h",

      palette="Reds_r", edgecolor="gray")



# Use the same x axis limits on all columns and add better labels

g.set(xlabel="Wins", ylabel="")



# Use semantically meaningful titles for the columns

titles = ["Total Wins", "Local Wins", "Visitor Wins"]



for ax, title in zip(g.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(False)

    ax.yaxis.grid(True)



sns.despine(left=True, bottom=True)
# Same again but for past 5 years

sns.set(style="whitegrid")



# Rearrange data

top_50 = wins_per_season[(wins_per_season['team'].isin(total_wins_sorted_desc[:50]))

                         &

                         (wins_per_season['season'].isin(df['season'].sort_values(ascending=False)[:5]))]

wps = top_50[['team','total_wins','local_wins','visitor_wins']]

wps = wps.groupby(['team'])['total_wins','local_wins','visitor_wins'].sum().reset_index()



# Make the PairGrid

g = sns.PairGrid(wps.sort_values("total_wins", ascending=False),

                 x_vars=wps.columns[1:], y_vars=["team"],

                 size=10, aspect=.25)



# Draw a dot plot using the stripplot function

g.map(sns.stripplot, size=10, orient="h",

      palette="Reds_r", edgecolor="gray")



# Use the same x axis limits on all columns and add better labels

g.set(xlabel="Wins", ylabel="")



# Use semantically meaningful titles for the columns

titles = ["Total Wins", "Local Wins", "Visitor Wins"]



for ax, title in zip(g.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(False)

    ax.yaxis.grid(True)



sns.despine(left=True, bottom=True)
# Looking at wins per season for the top 3, each dot is a season

top3 = wins_per_season[wins_per_season['team'].isin(['Real Madrid', 'Barcelona', 'Atletico de Madrid'])]

melt = pd.melt(top3[['team','local_wins','visitor_wins']], 'team', var_name='wins')

sns.swarmplot(x="wins", y="value", hue="team", data=melt)