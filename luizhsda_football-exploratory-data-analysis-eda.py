# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
## Loading events dataset
df_events = pd.read_csv("../input/events.csv")
## Loading ginf dataset, which has some important data to merge with event dataset
df_ginf = pd.read_csv("../input/ginf.csv")
## Selecting only the features I need. `id_odsp` serves as a unique identifier that will be used to 
## union the 2 datasets
df_ginf = df_ginf[['id_odsp', 'date', 'league', 'season', 'country']]
## Join the 2 datasets to add 'date', 'league', 'season', and 'country' information to the main dataset
df_events = df_events.merge(df_ginf, how='left')
## Filtering the seasons because some leagues have many more matches than others before season 2015
df_events = df_events[df_events['season'] >= 2015]
## Naming the leagues with their popular names, which will make thinks much clear for us
leagues = {'E0': 'Premier League', 'SP1': 'La Liga',
          'I1': 'Serie A', 'F1': 'League One', 'D1': 'Bundesliga'}
## Apply the mapping
df_events['league'] = df_events['league'].map(leagues)
## Events type 1
event_type_1 = pd.Series([
    'Announcement',
    'Attempt',
    'Corner',
    'Foul',
    'Yellow card',
    'Second yellow card',
    'Red card',
    'Substitution',
    'Free kick won',
    'Offside',
    'Hand ball',
    'Penalty conceded'], index=[[item for item in range(0, 12)]])

## Events type 2
event_type2 = pd.Series(['Key Pass', 'Failed through ball', 'Sending off', 'Own goal'],
                       index=[[item for item in range(12, 16)]])

## Match side
side = pd.Series(['Home', 'Away'], index=[[item for item in range(1, 3)]])

## Shot place
shot_place = pd.Series([
    'Bit too high', 
    'Blocked',
    'Bottom left corner',
    'Bottom right corner',
    'Centre of the goal',
    'High and wide',
    'Hits the bar',
    'Misses to the left',
    'Misses to the right',
    'Too high',
    'Top centre of the goal',
    'Top left corner',
    'Top right corner'
], index=[[item for item in range(1, 14)]])

## Outcome of shot
shot_outcome = pd.Series(['On target', 'Off target', 'Blocked', 'Hit the bar'],
                        index=[[item for item in range(1, 5)]])
## Location of shot
location = pd.Series([
    'Attacking half',
    'Defensive half',
    'Centre of the box',
    'Left wing',
    'Right wing',
    'Difficult angle and long range',
    'Difficult angle on the left',
    'Difficult angle on the right',
    'Left side of the box',
    'Left side of the six yard box',
    'Right side of the box',
    'Right side of the six yard box',
    'Very close range',
    'Penalty spot',
    'Outside the box',
    'Long range',
    'More than 35 yards',
    'More than 40 yards',
    'Not recorded'
], 
index=[[item for item in range(1, 20)]])

## Players' body part
bodypart = pd.Series(['right foot', 'left foot', 'head'], index=[[item for item in range(1, 4)]])

## Assist method
assist_method = pd.Series(['None', 'Pass', 'Cross', 'Headed pass', 'Through ball'],
                         index=[item for item in range(0, 5)])

## Situation
situation = pd.Series(['Open play', 'Set piece', 'Corner', 'Free kick'],
                     index=[item for item in range(1, 5)])
## Utility function to plot bar plots with similar configuration
def plot_barplot(data, x_ticks, x_labels, y_labels, title, color='muted', num=10, figsize=8):
    ## Set standard style as whitegrid (this also could be customized via param)
    sns.set_style("whitegrid")
    ## Set a figure with custom figsize
    plt.figure(figsize=(num, figsize))
    ## Plottin data
    ax = sns.barplot(x = [j for j in range(0, len(data))], y=data.values, palette=color)
    ## Setting ticks extracted from data indexes
    ax.set_xticks([j for j in range(0, len(data))])
    ## Set labels of the chart
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set(xlabel = x_labels, ylabel = y_labels, title = title)
    ax.plot();
    plt.tight_layout()
## Count of events occurecies
events_series = df_events['event_type'].value_counts()

## Plotting chart 
plot_barplot(events_series, event_type_1.values,
            "Event type", "Number of events", "Event types", 'OrRd_r', 10, 5)
## Filtering out dataframe to extract attemtps which resulted in goals
df_shot_places = df_events[(df_events['event_type'] == 1) & 
                           (df_events['is_goal'] == 1)]['shot_place'].value_counts()

## Plotting the chart
plot_barplot(df_shot_places, shot_place[[3, 4, 5, 13, 12]], 'Shot places', 'Number of events',
    'Shot places resulting in goals', 'BuGn_r', 8, 5)
## Copying original dataframe
df_shot_places_ed = df_events.copy()

## Grouping data by shot places
df_shot_places_ed = df_events.groupby('shot_place', as_index=False).count().sort_values('id_event',
                                                   ascending=False).dropna()

## Mapping dataframe index to shot places labels available in the dictionary file
df_shot_places_ed['shot_place'] = df_shot_places_ed['shot_place'].map(shot_place)

## Plotting the chart
plot_barplot(df_shot_places_ed['id_event'], df_shot_places_ed['shot_place'],
             'Shot places',
             'Number of events',
             'Shot places',
             'BuGn_r', 8, 5)
## Grouping attempts by team
grouping_by_offensive = df_events[df_events['is_goal']==1].groupby('event_team')

## Sorting the values
grouping_by_offensive = grouping_by_offensive.count().sort_values(by='id_event', ascending=False)[:10]
teams = grouping_by_offensive.index
scores = grouping_by_offensive['id_event']

## Plotting the teams
plot_barplot(scores, teams, 'Teams', '# of goals', 'Most offensive teams', 'GnBu_d', 8, 6)
## grouping by player when is goal
grouping_by_offensive_player = df_events[df_events['is_goal']==1].groupby('player')

## Couting and sorting the number of goals by player, then pick the top 10
grouping_by_offensive_player = grouping_by_offensive_player.count().sort_values(by='id_event',
                                                                                ascending=False)[:10]
## Extracting player names
players = grouping_by_offensive_player.index
## Extracting values (# of goals)
scores = grouping_by_offensive_player['id_event']

## Plotting the chart
plot_barplot(scores, players, 'Players', '# of score', 'Most offensive players', 'GnBu_d', 8, 6)
## Grouping by player when attempt == 1
grouped_by_player = df_events[df_events['event_type'] == 1].groupby('player').count()

## Grouping by player when is goal and attempt == 1
grouped_by_player_goals = df_events[(df_events['event_type'] == 1) &
                             (df_events['is_goal'] == 1)].groupby('player').count()

## Grouping by player when is not goal and attempt == 1
grouped_by_player_not_goals = df_events[(df_events['event_type'] == 1) &
                             (df_events['is_goal'] == 0)].groupby('player').count()

## Setting a threshold to filter out players with a small number of attempts, which can lead to a lack 
## of consistency in the final result
threshold = grouped_by_player['id_event'].std()

grouped_by_player_is_goal = df_events[df_events['is_goal'] == 1].groupby('player').count()
## filtering players that have at least more than average number of chances to skip noise
## For example, a player that has 2 attempts and 1 goal has a very high effectiveness, even though,
## this particular player has not created many chances for his team
grouped_by_player_is_goal_filtered = grouped_by_player_goals[grouped_by_player_goals['id_event'] > threshold]
grouped_by_players_not_goal_filtered = grouped_by_player_not_goals[grouped_by_player_not_goals['id_event'] > threshold]

## Total number of attemtps
total = grouped_by_players_not_goal_filtered['id_event'] + grouped_by_player_is_goal_filtered['id_event']

## Dividing the total of attempts by the attemtps which ended up in goals
result = total/grouped_by_player_is_goal_filtered['id_event']

## Dropping NaN values
result.dropna(inplace=True)

## Sorting results
sorted_results = result.sort_values(ascending=True)

## Extracting players names
players = sorted_results[:10].index
## Extracting value of effectiveness
effectiveness = sorted_results[:10]

## Plotting results
plot_barplot(effectiveness, players, 'Players', 
            '# of shots needed to score a goal',
            'Most effective players',
            'RdBu_r', 8, 6)
## Creating a dataframe with total of attempts and total goals
result_df = pd.DataFrame({'total': total.dropna(), 'is_goal': grouped_by_player_is_goal_filtered['id_event']})
## Sorting values by total
result_df.sort_values('total', ascending=False, inplace=True)

## Setting style to dark
sns.set(style="darkgrid")

## Creating figure
f, ax = plt.subplots(figsize=(10, 6))

## Plotting chart
sns.set_color_codes("pastel")
sns.barplot(x="total",
            y=result_df.index,
            data=result_df,
            label="# of attempts", color="b")

sns.set_color_codes("muted")
sns.barplot(x='is_goal',
            y=result_df.index, 
            data=result_df,
            label="# of goals", color="b")

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(ylabel="Players",
       xlabel="Number of goals x attempts", title='Player effectiveness')

each = result_df['is_goal'].values
the_total = result_df['total'].values
x_position = 50

for i in range(len(ax.patches[:15])):
    ax.text(ax.patches[i].get_width() - x_position, ax.patches[i].get_y() +.50, 
            str(round((each[i]/the_total[i])*100, 2))+'%')
    
sns.despine(left=True, bottom=True)
f.tight_layout()
## Filtering out events with time <= 15'
first_15 = df_events[df_events['time'] <= 15]
## Filtering out events with time between 75' and 90'
last_15 = df_events[(df_events['time'] >= 75) & (df_events['time'] <= 90)]

## Grouping by teams for the first 15'
top_10_scorer_first_15 = first_15[first_15['is_goal'] == 1].groupby('event_team').count().sort_values(by='id_event', ascending=False)

## Extracting teams from dataframe
teams = top_10_scorer_first_15.index[:10]
## Extracting number of goals 
scores = top_10_scorer_first_15['id_event'][:10]

## Plotting results
sns.set_style("whitegrid")
fig, axs = plt.subplots(ncols=2, figsize=(15, 6))
ax = sns.barplot(x = [j for j in range(0, len(scores))], y=scores.values, palette='Blues_d', ax=axs[0])
ax.set_xticks([j for j in range(0, len(scores))])
ax.set_xticklabels(teams, rotation=45)
ax.set(xlabel = 'Teams', ylabel = '# of goals', title = 'Most goals in the 1st 15 minutes');

## Grouping by last 15' scorers
top_10_scorer_last_15 = last_15[last_15['is_goal'] == 1].groupby('event_team').count().sort_values(by='id_event', ascending=False)[:10]

## Extracting the names of the teams
teams_last_15 = top_10_scorer_last_15.index[:10]
## Extracting the number of goals
scores_last_15 = top_10_scorer_last_15['id_event'][:10]

## Plottin the results
ax = sns.barplot(x = [j for j in range(0, len(scores_last_15))], y=scores_last_15.values, palette='Blues_d', ax=axs[1])
ax.set_xticks([j for j in range(0, len(scores_last_15))])
ax.set_xticklabels(teams_last_15, rotation=45)
ax.set(xlabel = 'Teams', ylabel = '# of goals', title = 'Most goals in the last 15 minutes');
fig.tight_layout()
## Last minute goals
last_minute_goals = df_events[(df_events['time'] >= 90) &
                             (df_events['time'] <= 95)]

## Last minute scores by team, league, opponent, and players
top_last_minute_scores = last_minute_goals[last_minute_goals['is_goal'] == 1].groupby(['event_team', 'league']).count().sort_values(by='id_event', ascending=False)
top_last_minute_scored = last_minute_goals[last_minute_goals['is_goal'] == 1].groupby('opponent').count().sort_values(by='id_event', ascending=False)
top_last_minute_scores_by_player = last_minute_goals[last_minute_goals['is_goal'] == 1].groupby('player').count().sort_values(by='id_event', ascending=False)

## Plotting the last minute goals by players
plot_barplot(top_last_minute_scores_by_player['id_event'][:10],
            top_last_minute_scores_by_player.index[:10],
            'Players',
            '# of goals',
            'Last minute goals by player',
            'Reds_r', 8, 5)
## Reseting index
top_last_minute_scores.reset_index(inplace=True)

## Filtering out dataframe by league
top_5_scorer_league_one = top_last_minute_scores[top_last_minute_scores['league'] == 'League One'][:5]
top_5_scorer_premier_league = top_last_minute_scores[top_last_minute_scores['league'] == 'Premier League'][:5]
top_5_scorer_la_liga = top_last_minute_scores[top_last_minute_scores['league'] == 'La Liga'][:5]
top_5_scorer_serie_a = top_last_minute_scores[top_last_minute_scores['league'] == 'Serie A'][:5]
top_5_scorer_bundesliga = top_last_minute_scores[top_last_minute_scores['league'] == 'Bundesliga'][:5]

## Plotting the chart
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
ax = sns.barplot(x = [j for j in range(0, len(top_5_scorer_league_one['id_event']))],
                 y=top_5_scorer_league_one['id_event'].values,
                 palette='Reds_r', ax=axs[0][0])
ax.set_xticks([j for j in range(0, len(top_5_scorer_league_one['id_event']))])
ax.set_xticklabels(top_5_scorer_league_one['event_team'], rotation=45)
ax.set(xlabel = 'Teams', ylabel = '# of goals', title = 'League One');

ax = sns.barplot(x = [j for j in range(0, len(top_5_scorer_premier_league['id_event']))],
                 y=top_5_scorer_premier_league['id_event'].values,
                 palette='Reds_r', ax=axs[0][1])
ax.set_xticks([j for j in range(0, len(top_5_scorer_premier_league['id_event']))])
ax.set_xticklabels(top_5_scorer_premier_league['event_team'], rotation=45)
ax.set(xlabel = 'Teams', ylabel = '# of goals', title = 'Premiere League');

ax = sns.barplot(x = [j for j in range(0, len(top_5_scorer_la_liga['id_event']))],
                 y=top_5_scorer_la_liga['id_event'].values,
                 palette='Reds_r', ax=axs[0][2])
ax.set_xticks([j for j in range(0, len(top_5_scorer_la_liga['id_event']))])
ax.set_xticklabels(top_5_scorer_la_liga['event_team'], rotation=45)
ax.set(xlabel = 'Teams', ylabel = '# of goals', title = 'La Liga');

ax = sns.barplot(x = [j for j in range(0, len(top_5_scorer_serie_a['id_event']))],
                 y=top_5_scorer_serie_a['id_event'].values,
                 palette='Reds_r', ax=axs[1][0])
ax.set_xticks([j for j in range(0, len(top_5_scorer_serie_a['id_event']))])
ax.set_xticklabels(top_5_scorer_serie_a['event_team'], rotation=45)
ax.set(xlabel = 'Teams', ylabel = '# of goals', title = 'Serie A');

ax = sns.barplot(x = [j for j in range(0, len(top_5_scorer_bundesliga['id_event']))],
                 y=top_5_scorer_bundesliga['id_event'].values,
                 palette='Reds_r', ax=axs[1][1])
ax.set_xticks([j for j in range(0, len(top_5_scorer_bundesliga['id_event']))])
ax.set_xticklabels(top_5_scorer_bundesliga['event_team'], rotation=45)
ax.set(xlabel = 'Teams', ylabel = '# of goals', title = 'Bundesliga');

axs[1][2].axis('off')

fig.tight_layout()
## Copying original dataset
df_shot_outcome = df_events.copy()
## Mapping shot outcomes to original labels
shot_outcomes = {1: 'On Target', 2: 'Off Target', 3: 'Blocked', 4: 'Hit the Bar'}
df_shot_outcome['shot_outcome'] = df_shot_outcome['shot_outcome'].map(shot_outcome)

## Plotting the chart
g = sns.FacetGrid(df_shot_outcome, col="shot_outcome")
g = g.map(plt.hist, 'league', color="r")
g.set_xticklabels(rotation=45)
g.set_ylabels('# of attempts');
## Total number of attempts on target
total_events = df_events[(df_events['event_type'] == 1) &
         ((df_events['shot_outcome'] == 1) |
         (df_events['shot_outcome'] == 2))].groupby('player').count()
         
## Using the threshold to filter out players with fewer events
total_events = total_events[total_events['id_event'] > threshold]

## How many of those attempt events were actually on target
on_target_events = df_events[(df_events['event_type'] == 1) &
         (df_events['shot_outcome'] == 1)].groupby('player').count()

threshold = 100

## Filtering out the result
total_events = total_events[total_events['id_event'] > threshold]

## Dividing the number of attemps by the attempts on target
on_target = on_target_events['id_event']/total_events[total_events.index.isin(on_target_events.index)]['id_event']

## Sorting the result to find the most 'accurate' players
most_accurate_players = on_target.dropna().sort_values(ascending=False)[:10]
less_accurate_players = on_target.dropna().sort_values()[:10]

sns.set_style("whitegrid")
## Plot the most accurate players chart
fig, axs = plt.subplots(ncols=2, figsize=(15, 6))
ax = sns.barplot(x = [j for j in range(0, len(most_accurate_players))], y=most_accurate_players.values, palette='Blues_d', ax=axs[0])
ax.set_xticks([j for j in range(0, len(most_accurate_players))])
ax.set_xticklabels(most_accurate_players.index, rotation=45)
ax.set(xlabel = 'Players', ylabel = '# of attempts to target the goal', title = 'Most accurate players');

## Plot the less accurate players chart
ax = sns.barplot(x = [j for j in range(0, len(less_accurate_players))], y=less_accurate_players.values, palette='Blues_d', ax=axs[1])
ax.set_xticks([j for j in range(0, len(less_accurate_players))])
ax.set_xticklabels(less_accurate_players.index, rotation=45)
ax.set(xlabel = 'Players', ylabel = '# of attempts to target the goal', title = 'Less accurate players');
import cufflinks as cf
cf.go_offline()
## Grouping players who had their shots blocked
players_blocked = df_events[df_events['shot_outcome'] == 3].groupby('player').count().sort_values('id_event',
                                                                                                  ascending=False)
## Grouping players who have blocked the most shots
players_blocked_the_most = df_events[df_events['shot_outcome'] == 3].groupby('player2').count().sort_values('id_event',
                                                                                                            ascending=False)

## Plotting the top 10 players that have the most shots blocked
players_blocked[:10]['id_event'].iplot(kind='bar', yTitle='# of events',
                                         title='Players that have their most shots blocked',
             filename='cufflinks/categorical-bar-chart', color='#27AE60')
## Players that blocked the most attempts
players_blocked_the_most[:10]['id_event'].iplot(kind='bar', yTitle='# of events',
                                         title='Players who blocked the most shots',
             filename='cufflinks/categorical-bar-chart', color='#00CEC9')
## Most goals from outside the box
long_range_goals = df_events[(df_events['location'] == 15) & 
         (df_events['is_goal'] == 1)].groupby('player').count().sort_values('id_event',
                                                                           ascending=False)[:10]
## Long range attempts (goals and not goals)
long_range_attempts = df_events[(df_events['location'] == 15)].groupby('player').count().sort_values('id_event',
                                                                           ascending=False)
## Most goals from the long range
long_range_goals['id_event'].iplot(kind='bar', yTitle='# of events',
                                         title='Top long range scorers',
             filename='cufflinks/categorical-bar-chart', color='#2D3436')
print('{} # of attempts: {}'.format('Yaya Toure', long_range_attempts.iloc[long_range_attempts.index.get_loc('yaya toure')]['id_event']))
## Copying original dataset
df_body_part = df_events.copy()

## Mapping dataframe bodypart to provided labels
df_body_part['bodypart'] = df_body_part['bodypart'] .map(bodypart)

## Filtering only the goal events
df_body_part = df_body_part[df_body_part['is_goal'] == 1]

## Setting custom style and plotting the chart
sns.color_palette("bright")
sns.set(font_scale=1.0)
sns.set_style("white")
g = sns.FacetGrid(df_body_part, col="bodypart", col_wrap=4, sharey=True)
g = g.map(plt.hist, 'league', bins=10, color="r")
g.set_xticklabels(rotation=45)
g.set_xlabels('League')
g.set_ylabels('# of events');
## Copying the original dataset
df_goals = df_events.copy()

## Filtering goals events
df_goals = df_goals[df_goals['is_goal'] == 1]

## Goals with right foot
right_foot = df_goals[df_goals['bodypart'] == 1].groupby('player').count().sort_values('id_event',
                                                                          ascending=False)[:5]

## Goals with left foot
left_foot = df_goals[df_goals['bodypart'] == 2].groupby('player').count().sort_values('id_event',
                                                                           ascending=False)[:5]

## Goals with head
head = df_goals[df_goals['bodypart'] == 3].groupby('player').count().sort_values('id_event',
                                                                           ascending=False)[:5]

## Most goals by bodypart
fig, axs = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)
ax = sns.barplot(x = [j for j in range(0, len(right_foot['id_event']))],
                 y=right_foot['id_event'].values, palette='Blues_r', ax=axs[0])
ax.set_xticks([j for j in range(0, len(right_foot['id_event']))])
ax.set_xticklabels(right_foot['event_team'].index, rotation=45)
ax.set(xlabel = 'Players', ylabel = '# of goals', title = 'Right foot');

ax = sns.barplot(x = [j for j in range(0, len(left_foot['id_event']))],
                 y=left_foot['id_event'].values, palette='Blues_r', ax=axs[1])
ax.set_xticks([j for j in range(0, len(left_foot['id_event']))])
ax.set_xticklabels(left_foot['event_team'].index, rotation=45)
ax.set(xlabel = 'Players', ylabel = '# of goals', title = 'Left foot');

ax = sns.barplot(x = [j for j in range(0, len(head['id_event']))],
                 y=head['id_event'].values, palette='Blues_r', ax=axs[2])
ax.set_xticks([j for j in range(0, len(head['id_event']))])
ax.set_xticklabels(head['event_team'].index, rotation=45)
ax.set(xlabel = 'Players', ylabel = '# of goals', title = 'Head');
## Grouping data by assists
top_assists = df_events[(df_events['is_goal'] == 1) &
         (df_events['player2'] != np.nan)].groupby('player2').count().sort_values('id_event',
                                                                   ascending=False)[:10]

## Plotting the players with the most assists
top_assists['id_event'][:10].iplot(kind='bar', yTitle='# of assists', title='Top assists',
             filename='cufflinks/categorical-bar-chart', color='#C70039')
## Setting style
sns.set_style("whitegrid")

## Setting figure
plt.figure(figsize=(12, 5))

## Plotting data
ax = sns.stripplot(top_assists.sort_values('id_event').index, top_assists.sort_values('id_event')['id_event'].values, 
                palette='Blues', size=15, jitter=True,
                linewidth=1)
ax.set(ylabel="# of assists",
       xlabel="Players", title='Players with the most assists');
ax.set_xticklabels(top_assists.sort_values('id_event').index, rotation=45);
## Copying original dataet
df_events_distribution = df_events.copy()

## Filtering the goals events
df_events_distribution = df_events_distribution[df_events_distribution['is_goal']==1]
## Converting the data column data datetime
df_events_distribution['date'] = pd.to_datetime(df_events_distribution['date'])

## Extracting day, month, and year from provided date and creating a exclusive column for each
df_events_distribution['day'] = df_events_distribution['date'].apply(lambda x: x.day)
df_events_distribution['month'] = df_events_distribution['date'].apply(lambda x: x.month)
df_events_distribution['year'] = df_events_distribution['date'].apply(lambda x: x.year)

## Extracting only months that have events (Some months don't have occorencies beacause of the season break)
sorted_indexes = sorted(df_events_distribution['month'].unique())

## Month labels
months =['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

## Overall mean by month
total_mean = df_events_distribution.groupby('month').count().mean()['id_event']

## Plotting the results
sns.set_style('dark')
plt.figure(figsize=(15, 5))
plt.bar(sorted_indexes, 
        df_events_distribution.groupby('month').count()['id_event'], alpha=0.6,
       color='skyblue')
plt.plot(sorted_indexes, 
        df_events_distribution.groupby('month').count()['id_event'], color='darkred', marker='o',
        linewidth=4, markersize=10)
plt.axvspan(6, 7, alpha=0.5, color='azure')
plt.xticks([item for item in range(1, 13)], months);
plt.ylabel('# of gols')
plt.title('Goals distribution by month in Europe');
## Goal distribution by each league
premier_league_events = df_events_distribution[df_events_distribution['league'] == 'Premier League']
la_liga_events = df_events_distribution[df_events_distribution['league'] == 'La Liga']
league_one_events = df_events_distribution[df_events_distribution['league'] == 'League One']
serie_a_events = df_events_distribution[df_events_distribution['league'] == 'Serie A']
bundesliga_events = df_events_distribution[df_events_distribution['league'] == 'Bundesliga']

## Mean values for each league
pl_mean = premier_league_events.groupby('month').count().mean()['id_event']
ll_mean = la_liga_events.groupby('month').count().mean()['id_event']
lo_mean = league_one_events.groupby('month').count().mean()['id_event']
sa_mean = serie_a_events.groupby('month').count().mean()['id_event']
bl_mean = bundesliga_events.groupby('month').count().mean()['id_event']

## Random color to use in charts
colors = ['olivedrab', 'seagreen', 'lightseagreen', 'paleturquoise',
'darkcyan', 'deepskyblue', 'royalblue', 'darkslategray',
'cadetblue', 'dodgerblue', 'lightskyblue', 'teal']

## Plotting the distribution by league
sns.set_style('ticks')
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))

axs[0][0].plot(premier_league_events.groupby('month').count()['id_event'], lw=4, color=random.choice(colors));
## premier league mean
axs[0][0].axhline(y=pl_mean, color='r', linestyle='-', alpha=0.5)
axs[0][0].set_xlim(1, 12);
axs[0][0].set_xticks([item for item in range(1, 13)])
axs[0][0].set_xticklabels(months);
axs[0][0].set_ylabel('# of goals')
axs[0][0].set_title('Premier League goals distribution')

axs[0][1].plot(la_liga_events.groupby('month').count()['id_event'], lw=4, color=random.choice(colors));
## la liga mean
axs[0][1].axhline(y=ll_mean, color='r', linestyle='-', alpha=0.5)
axs[0][1].set_xlim(1, 12);
axs[0][1].set_xticks([item for item in range(1, 13)])
axs[0][1].set_xticklabels(months);
axs[0][1].set_ylabel('# of goals')
axs[0][1].set_title('La Liga goals distribution')

axs[1][0].plot(league_one_events.groupby('month').count()['id_event'], lw=4, color=random.choice(colors));
## league one mean
axs[1][0].axhline(y=lo_mean, color='r', linestyle='-', alpha=0.5)
axs[1][0].set_xlim(1, 12);
axs[1][0].set_xticks([item for item in range(1, 13)])
axs[1][0].set_xticklabels(months);
axs[1][0].set_ylabel('# of goals')
axs[1][0].set_title('League One goals distribution')

axs[1][1].plot(serie_a_events.groupby('month').count()['id_event'], lw=4, color=random.choice(colors));
## serie a league
axs[1][1].axhline(y=sa_mean, color='r', linestyle='-', alpha=0.5)
axs[1][1].set_xlim(1, 12);
axs[1][1].set_xticks([item for item in range(1, 13)])
axs[1][1].set_xticklabels(months);
axs[1][1].set_ylabel('# of goals')
axs[1][1].set_title('Serie A goals distribution')

axs[2][0].plot(bundesliga_events.groupby('month').count()['id_event'], lw=4, color=random.choice(colors));
## bundesliga mean
axs[2][0].axhline(y=bl_mean, color='r', linestyle='-', alpha=0.5)
axs[2][0].set_xlim(1, 12);
axs[2][0].set_xticks([item for item in range(1, 13)])
axs[2][0].set_xticklabels(months);
axs[2][0].set_ylabel('# of goals')
axs[2][0].set_title('Bundesliga goals distribution')

axs[2][1].axis('off');
plt.tight_layout()
## Plotly version of the same visual

## Grouping event by month
grouped_monthly_distribution = df_events_distribution.groupby('month').count()

## Excluding months with no events
active_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

## Plotting La liga data
la_liga = go.Scatter(
                x=active_months,
                y=la_liga_events.groupby('month').count()['id_event'].values,
                name = "La Liga",
                line = dict(color = '#FF5733'),
                opacity = 0.8)

## Plot Bundesliga data
bundesliga = go.Scatter(
                x=active_months,
                y=bundesliga_events.groupby('month').count()['id_event'].values,
                name = "Bundesliga",
                line = dict(color = '#FFC300'),
                opacity = 0.8)

## Plotting League One data
league_one = go.Scatter(
                x=active_months,
                y=league_one_events.groupby('month').count()['id_event'].values,
                name = "League One",
                line = dict(color = '#17BECF'),
                opacity = 0.8)

## Plotting Serie A data
serie_a = go.Scatter(
                x=active_months,
                y=serie_a_events.groupby('month').count()['id_event'].values,
                name = "Serie A",
                line = dict(color = '#75FF33'),
                opacity = 0.8)

## Plotting Premier League data
premier_league = go.Scatter(
                x=active_months,
                y=premier_league_events.groupby('month').count()['id_event'].values,
                name = "Premier League",
                line = dict(color = '#C70039'),
                opacity = 0.8)

data = [la_liga, bundesliga, league_one, serie_a, premier_league]

layout = dict(
    title = "Goals distribution by month",
     xaxis = dict(
        showticklabels=True)
)

fig = dict(data=data, layout=layout)
py.iplot(fig, filename = "Monthly goals distribution in Europe")
## Copying original dataset
df_distribution_minutes = df_events.copy()

## Goals distribution by minutes
df_distribution_minutes = df_distribution_minutes[(df_distribution_minutes['is_goal'] == 1) &
                                                ((df_distribution_minutes['time'] >=0 &
                                                (df_distribution_minutes['time'] <=90)))]
## Couting the goals by minutes
group_goals_by_minutes = df_distribution_minutes.groupby('time').count()

## Plotting using plotly
group_goals_by_minutes['id_event'].iplot(kind='bar', yTitle='# of goals',
                                         title='Goals distribution by minutes',
             filename='cufflinks/categorical-bar-chart', color='#255818')
## Copying original data
df_event_type_distribution_minutes = df_events.copy()

## Grouped data function for events distribution in matches
def get_event_type_df(event_type):
    """ Function to return grouped data for a specific event type, passed as parameter
        get_event_type_df(event_type) -> dataframe 
    """
    return df_event_type_distribution_minutes[df_event_type_distribution_minutes['event_type'] == event_type].groupby('time').count()

## Setting dark style
sns.set_style('darkgrid')
fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(12, 15), sharex=True)

## Custom colors
colors = ['darkgreen', 'royalblue', 'dimgray', 'yellow', 'yellow', 'darkred']
## Event titles
titles = ['Attempt', 'Corner', 'Foul', 'Yellow Card', '2nd Yellow Card', 'Red Card']

## Plotting events
for item in range(0, 6):
    ax = sns.barplot(x = [j for j in range(0, len(get_event_type_df(item+1)['id_event']))],
                 y=get_event_type_df(item+1)['id_event'].values, color=colors[item], ax=axs[item])
    ax.set_xticks([j for j in range(0, len(get_event_type_df(item+1)['id_event']), 5)])
    ax.set_xticklabels(([(str(i) + "'") for i in range(0, len(get_event_type_df(item+1)['id_event']), 5)]))
    ax.set(xlabel = 'Minute of the match', ylabel = '# of events', title = titles[item]);

plt.tight_layout()
## Goals distribution by year
goals_dist_year = df_events_distribution.groupby(['time', 'season'], as_index=False).count()

## Breaking down by season
season_2015 = goals_dist_year[goals_dist_year['season'] == 2015]
season_2016 = goals_dist_year[goals_dist_year['season'] == 2016]
## The dataset does not contain all the data for the 2017 season. For this reason, we're excluding it 
## to keep the consistency
season_2017 = goals_dist_year[goals_dist_year['season'] == 2017]

## Plotting data

fig = {
    'data': [
        {'x': season_2015['time'].values,
         'y': season_2015['id_event'].values, 'mode': 'markers', 'name': 'Season 2015'},
        {'x': season_2016['time'].values,
         'y': season_2016['id_event'].values, 'mode': 'markers', 'name': 'Season 2016'}        
    ],
    'layout': {
        'xaxis': {'title': 'Minute of the match'},
        'yaxis': {'title': "# of goals"}
    }
}
py.iplot(fig, filename='cufflinks/multiple-scatter')
## Grouping by assist method
which_assist = df_events[df_events['is_goal'] == 1].groupby('assist_method').count().sort_values('id_event',
                                                                                 ascending=False)
## Mapping index numbers to labels
which_assist.index = assist_method

## Plotting data
which_assist['id_event'].iplot(kind='bar', yTitle='# of goals',
                                         title='Goals distribution by minutes',
             filename='cufflinks/categorical-bar-chart', color='#31255A')
## Grouping data by sitation where a goal occurred
which_situation = df_events[df_events['is_goal'] == 1].groupby('situation').count().sort_values('id_event',
                                                                                 ascending=False)
## Mapping index to labels
which_situation.index = situation

## Plotting data
which_situation['id_event'].iplot(kind='bar', yTitle='# of goals',
                                         title='Situation where goals occurred',
             filename='cufflinks/categorical-bar-chart', color='#07B0E8')
## Filtering data by fouls
df_fouls_which_league = df_events[df_events['event_type'] == 3]

## Grouping by league
df_fouls_which_league = df_fouls_which_league.groupby('league').count().sort_values('id_event',
                                                           ascending=True)

## Plotting data
df_fouls_which_league['id_event'].iplot(kind='bar', yTitle='# of fouls',
                                         title='Number of fouls per league',
             filename='cufflinks/categorical-bar-chart', color='#07B0E8')
# Filtering data by cards
df_cards = df_events[(df_events['event_type'] == 4) |
                     (df_events['event_type'] == 5) |
                     (df_events['event_type'] == 6)]

## Grouping data
grouped_card_events = df_cards.groupby(['league', 'event_type'],
                                       as_index=False).count().sort_values('id_event',
                                                                           ascending=True)
## Creating a new dataframe to group the results
cards = pd.DataFrame()

## Results for each card type
yellow_cards = grouped_card_events[grouped_card_events['event_type'] == 4][['league', 'id_event']].sort_index()
second_yellow_card = grouped_card_events[grouped_card_events['event_type'] == 5][['league', 'id_event']].sort_index()
red_cards = grouped_card_events[grouped_card_events['event_type'] == 6][['league', 'id_event']].sort_index()

## Creating dataframe column to each result
cards['Yellow card'] = yellow_cards['id_event'].values
cards['Second yellow card'] = second_yellow_card['id_event'].values
cards['Red card'] = red_cards['id_event'].values

## Mapping index to labels
cards.index = yellow_cards['league'].values

## Plotting the results
cards.iplot(kind='barh', yTitle='# of cards',
                                         title='Number of cards per league',
             filename='cufflinks/categorical-bar-chart', colors = ['#FCEA53', '#FEE300',
                                                                  '#DF0505'])
