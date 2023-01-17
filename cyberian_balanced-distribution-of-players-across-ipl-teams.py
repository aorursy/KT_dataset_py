import matplotlib as mpl

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from matplotlib.patches import Circle

from matplotlib.patheffects import withStroke

from sklearn.cluster import MiniBatchKMeans

import random

from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.simplefilter('ignore')
matches = pd.read_csv('../input/matches.csv')

deliveries = pd.read_csv('../input/deliveries.csv')
chk = deliveries.merge(matches, left_on='match_id', right_on='id', how='left')

print(np.sum(np.isnan(chk['win_by_runs'])))

chk = deliveries.groupby(['match_id', 'inning'])['over'].aggregate(np.max)

print(sum(chk>20)) #each inning is at max 20 overs
chk = deliveries.groupby(['match_id', 'inning', 'over'])['ball', 'wide_runs', 'noball_runs'].aggregate([np.sum, np.max])

chk.drop(labels=[('ball', 'sum'), ('wide_runs', 'amax'), ('noball_runs', 'amax')], inplace=True, axis=1)

print(sum(chk['ball']['amax'] - chk['wide_runs']['sum'] - chk['noball_runs']['sum'] > 6))
print(chk[(chk['ball']['amax'] - chk['wide_runs']['sum'] - chk['noball_runs']['sum'] >6)])
chk = deliveries.groupby(['match_id', 'inning', 'over', 'ball'])['batting_team'].aggregate(np.count_nonzero)

chk[(chk>1)]
chk = pd.DataFrame(chk[(chk>1)]).reset_index()[['match_id', 'inning', 'over', 'ball']]

duplicates = deliveries.merge(chk, how='inner',on=['match_id', 'inning', 'over', 'ball'])

tenthballrecords = duplicates.drop_duplicates(subset=['match_id', 'inning', 'over', 'ball'], keep='last')

tenthballrecords.loc[:, 'ball'] = 10

deliveries = deliveries.drop_duplicates(subset=['match_id', 'inning', 'over', 'ball'], keep='first')

deliveries = deliveries.append(tenthballrecords).sort_values(by=['match_id', 'inning', 'over', 'ball'])
chk = deliveries.groupby(['match_id', 'inning', 'over', 'ball'])['batting_team'].aggregate(np.count_nonzero)

sum(chk>1)==0 #True now
chk = deliveries.groupby(['match_id', 'inning'])['batsman'].nunique()

print(sum(chk>11)==0) #all good. Max 11 players batting per team

chk = deliveries.groupby(['match_id', 'inning'])['bowler'].nunique()

print(sum(chk>11)==0) #all good. Max 11 players bowler per team
matches.replace(to_replace='Deccan Chargers', value='Sunrisers Hyderabad', inplace=True)

deliveries.replace(to_replace='Deccan Chargers', value='Sunrisers Hyderabad', inplace=True)
teamdict = dict({'Chennai Super Kings': 'CSK', 'Delhi Daredevils': 'DD', 'Gujarat Lions': 'GL', 'Kings XI Punjab': 'KXIP', 'Kochi Tuskers Kerala': 'KTK', 'Kolkata Knight Riders': 'KKR', 'Mumbai Indians': 'MI', 'Pune Warriors': 'PW', 'Rajasthan Royals': 'RR', 'Rising Pune Supergiants': 'RPG', 'Royal Challengers Bangalore': 'RCB', 'Sunrisers Hyderabad': 'SRH'})

winners = matches.groupby('winner')['id'].aggregate(len).sort_values(ascending=False)

teams = [teamdict[t] for t in winners.index]

wins = winners.values

plot = plt.figure()

plt.bar(range(len(teams)), wins)

plt.gca().set_xlabel('Team')

plt.gca().set_ylabel('Wins')

plt.title('Wins by team')

plt.xticks(range(len(teams)), teams)

x = plt.gca().xaxis

# rotate the tick labels for the x axis

for item in x.get_ticklabels():

    item.set_rotation(45)
winners = pd.DataFrame(matches.groupby('winner')['id'].aggregate({'wins':len}))

played = pd.DataFrame(matches.groupby('team1')['id'].aggregate({'played':len}) + matches.groupby('team2')['id'].aggregate({'played':len}), columns=['played'])

winners = played.merge(winners, how='inner', left_index=True, right_index=True)

winners['win_pct'] = np.round((winners['wins']/winners['played']) * 100, decimals=0)

winners.sort_values('win_pct', ascending=False, inplace=True)

teams = [teamdict[t] for t in winners.index]



#Let's also overlay the average runs scored per match by team over the win % graph.

runs = deliveries.groupby('batting_team')['total_runs'].aggregate({'runs':np.sum})

runs = runs.merge(played, how='inner', left_index=True, right_index=True)

runs = runs['runs']/runs['played']

runs = [runs[idx] for idx in winners.index] #sort runs in the same order as the winners data



fig, ax1 = plt.subplots()

#plt.rcParams["figure.figsize"] = [9, 4]

ax1.bar(range(len(teams)), winners['win_pct'], label='Win %')

ax1.set_xlabel('Team')

ax1.set_ylabel('Win %')

plt.xticks(range(len(teams)), teams)



ax2 = ax1.twinx()

ax2.plot(range(len(teams)), runs, '-xr')

ax1.plot(np.nan, '-r', label='Avg. Runs') #workaround to get the legend to display both graphs

ax1.legend(loc=0)

ax2.set_ylabel('Avg. runs per match')

ax2.set_ylim(0, 175)



plt.title('Win % by team');
wkts = deliveries.groupby('bowling_team')['player_dismissed'].aggregate({'wickets':'count'})

wkts = wkts.merge(played, how='inner', left_index=True, right_index=True)

wkts = wkts['wickets']/wkts['played']

wkts = [wkts[idx] for idx in winners.index] #sort wickets in the same order as the winners data



fig, ax1 = plt.subplots()

#plt.rcParams["figure.figsize"] = [9, 4]

ax1.bar(range(len(teams)), winners['win_pct'], label='Win %')

ax1.set_xlabel('Team')

ax1.set_ylabel('Win %')

plt.xticks(range(len(teams)), teams)



ax2 = ax1.twinx()

ax2.plot(range(len(teams)), wkts, '-xr')

ax1.plot(np.nan, '-r', label='Avg. Wkts') #workaround to get the legend to display both graphs

ax1.legend(loc=0)

ax2.set_ylabel('Avg. wkts per match')

ax2.set_ylim(0, 7)



plt.title('Win % by team');
avg_runs_per_over = deliveries.groupby('over')['total_runs'].aggregate({'runs':'sum'})

overs_count = deliveries.groupby(['match_id', 'inning', 'over'])['ball'].aggregate({'ball':'max'}).reset_index().groupby('over')['match_id'].aggregate({'count':'count'})

avg_runs_per_over = avg_runs_per_over.merge(overs_count, how='inner', left_index=True, right_index=True)

avg_runs_per_over['rpo'] = np.round(avg_runs_per_over['runs']/avg_runs_per_over['count'], 2)

#Above gives us the rpo for each over across all matches

#Let's get a violin plot for number of runs per over.

runs_per_over = deliveries.groupby(['match_id', 'inning', 'over'])['total_runs'].aggregate({'runs': 'sum'}).reset_index().drop(labels=['match_id', 'inning'], axis=1)

data = [runs_per_over[(runs_per_over['over']==o)]['runs'] for o in np.unique(runs_per_over['over'])]

plt.rcParams["figure.figsize"] = [15, 4];

fig = plt.figure()

plt.violinplot(data, showmeans=True);

plt.xticks(range(len(data)+1));

plt.title('Distribution of runs per over');

plt.xlabel('Over number');

plt.ylabel('Runs scored');
def circle(x, y, radius=0.5, clr='black'):

    from matplotlib.patches import Circle

    from matplotlib.patheffects import withStroke

    circle = Circle((x, y), radius, clip_on=False, zorder=10, linewidth=1,

                    edgecolor=clr, facecolor=(0, 0, 0, .0125),

                    path_effects=[withStroke(linewidth=5, foreground='w')])

    plt.gca().add_artist(circle)





def text(x, y, text, clr='blue'):

    plt.gca().text(x, y, text, backgroundcolor="white",

            ha='center', va='top', weight='bold', color=clr)



#Now, how about running a k-means cluster to group the overs into 3 clusters based on average runs per over and see if it correlates with our observations above.

from sklearn.cluster import MiniBatchKMeans

kmodel = MiniBatchKMeans(n_clusters=3, random_state=0).fit(avg_runs_per_over['rpo'].values.reshape(-1, 1))

cluster_colours = ['#4EACC5', '#FF9C34', '#4E9A06']

clusters = kmodel.predict(avg_runs_per_over['rpo'].reshape(-1,1))

colours = [cluster_colours[clusters[o]] for o in range(len(clusters))]

fig = plt.figure()

plt.bar(avg_runs_per_over.index, avg_runs_per_over['rpo'], color=colours, label='Avg. runs in each over');

plt.title('k-means cluster with 3 clusters based on average runs in each over');

plt.xlabel('Over number');

plt.ylabel('Avg. runs in over');

plt.xticks(range(len(clusters)+1));

circle(6, -0.4, clr='blue')

text(6, -1, 'End of powerplay', clr='blue')
fours = deliveries[(deliveries['batsman_runs']==4)]

sixes = deliveries[(deliveries['batsman_runs']==6)]



fours = fours.groupby('batting_team')['batsman_runs'].aggregate({'count':'count'})

sixes = sixes.groupby('batting_team')['batsman_runs'].aggregate({'count':'count'})



fours = fours.merge(played, how='inner', left_index=True, right_index=True)

sixes = sixes.merge(played, how='inner', left_index=True, right_index=True)

fours = fours['count']/fours['played']

sixes = sixes['count']/sixes['played']





fours = [fours[idx] for idx in winners.index] #sort fours in the same order as the winners data

sixes = [sixes[idx] for idx in winners.index] #sort sixes in the same order as the winners data



fig, ax1 = plt.subplots()

#plt.rcParams["figure.figsize"] = [9, 4]

ax1.bar(range(len(teams)), winners['win_pct'], label='Win %')

ax1.set_xlabel('Team')

ax1.set_ylabel('Win %')

plt.xticks(range(len(teams)), teams)



ax2 = ax1.twinx()

ax2.plot(range(len(teams)), fours, '-xr')

ax2.plot(range(len(teams)), sixes, '-xg')

ax2.plot(range(len(teams)), np.array(fours)+np.array(sixes), '-xy')

ax1.plot(np.nan, '-r', label='# of 4s') #workaround to get the legend to display both graphs

ax1.plot(np.nan, '-g', label='# of 6s') #workaround to get the legend to display both graphs

ax1.plot(np.nan, '-y', label='# of 4s+6s') #workaround to get the legend to display both graphs

ax1.legend(loc=0)

ax2.set_ylabel('# of 4s and 6s')



plt.title('Win % by team vs. # of 4s/6s hit');
fours = deliveries[(deliveries['batsman_runs']==4)].merge(matches, left_on=['match_id', 'batting_team'], right_on=['id', 'winner'], how='left')

sixes = deliveries[(deliveries['batsman_runs']==6)].merge(matches, left_on=['match_id', 'batting_team'], right_on=['id', 'winner'], how='left')

fours_lost = fours[pd.isnull(fours['winner'])]

sixes_lost = sixes[pd.isnull(sixes['winner'])]

fours_match_count = fours_lost.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})

sixes_match_count = sixes_lost.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})

fours_lost = fours_lost.groupby('batting_team')['batsman_runs'].aggregate({'fours':'count'})

sixes_lost = sixes_lost.groupby('batting_team')['batsman_runs'].aggregate({'sixes':'count'})



fours_lost = fours_lost.merge(pd.DataFrame(fours_match_count), how='inner', left_index=True, right_index=True)

sixes_lost = sixes_lost.merge(pd.DataFrame(sixes_match_count), how='inner', left_index=True, right_index=True)

fours_lost['4s_per_match'] = fours_lost['fours']/fours_lost['played']

sixes_lost['6s_per_match'] = sixes_lost['sixes']/sixes_lost['played']

data_lost = pd.merge(fours_lost, sixes_lost, left_index=True, right_index=True, how='inner')

data_lost.sort_index(inplace=True)



fours_won = fours[~pd.isnull(fours['winner'])]

sixes_won = sixes[~pd.isnull(sixes['winner'])]

fours_match_count = fours_won.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})

sixes_match_count = sixes_won.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})

fours_won = fours_won.groupby('batting_team')['batsman_runs'].aggregate({'fours':'count'})

sixes_won = sixes_won.groupby('batting_team')['batsman_runs'].aggregate({'sixes':'count'})



fours_won = fours_won.merge(pd.DataFrame(fours_match_count), how='inner', left_index=True, right_index=True)

sixes_won = sixes_won.merge(pd.DataFrame(sixes_match_count), how='inner', left_index=True, right_index=True)

fours_won['4s_per_match'] = fours_won['fours']/fours_won['played']

sixes_won['6s_per_match'] = sixes_won['sixes']/sixes_won['played']

data_won = pd.merge(fours_won, sixes_won, left_index=True, right_index=True, how='inner')

data_won.sort_index(inplace=True)



fig = plt.figure()

plt.bar(np.array(range(len(data_lost.index)))-0.2, data_lost['4s_per_match'], color='blue', width=0.2, label='Avg. 4s in losses')

plt.bar(np.array(range(len(data_lost.index)))-0.2, data_lost['6s_per_match'], color='red', width=0.2, label='Avg. 6s in losses', bottom=data_lost['4s_per_match'])

plt.bar(np.array(range(len(data_won.index))), data_won['4s_per_match'], color='grey', width=0.2, label='Avg. 4s in wins')

plt.bar(np.array(range(len(data_won.index))), data_won['6s_per_match'], color='green', width=0.2, label='Avg. 6s in wins', bottom=data_won['4s_per_match'])

plt.xlabel('Team')

plt.ylabel('Avg. # of 4s and 6s in lost and won matches')

plt.xticks(range(len(data_lost.index)), [teamdict[t] for t in data_lost.index])

plt.yticks([3*n for n in range(10)])

plt.legend(frameon=False, loc=9)

plt.title('Avg. # of 4s/6s in lost and won matches');
matches_2016 = matches[(matches['season']==2016)]

deliveries_2016 = deliveries.merge(matches_2016, left_on='match_id', right_on='id', how='inner')

bowler_list = np.unique(deliveries_2016['bowler'])

batsman_list = np.unique(deliveries_2016['batsman'])

player_list = pd.DataFrame(np.union1d(bowler_list, batsman_list), columns=['player'])

#Let's now add features to the player dataframe

#Can the player bowl?

player_list = player_list.merge(pd.DataFrame(sorted(zip(bowler_list, np.ones(len(bowler_list)))), columns=['player', 'can_bowl']), how='left', left_on='player', right_on='player')

player_list.loc[(pd.isnull(player_list['can_bowl'])), 'can_bowl'] = 0

player_list.set_index('player', inplace=True)

#For the rest of the features, we follow the rule that better performance implies higher value for the feature. Poorer performance should result in lower value for the feature.

#This means that batsman scoring runs would be a regular feature, whereas the number of times he got out would need to be made a reciprocal of itself in the feature.

#Similarly, for bowling performance, wickets is a regular feature, whereas wides needs to be made a reciprocal of itself.

#The reason for this is to ensure the clustering works consistently on all features.

#how many runs in total did the player score

player_list = player_list.merge(deliveries_2016.groupby('batsman')['batsman_runs'].aggregate({'runs':'sum'}), how='left', left_index=True, right_index=True)

player_list.loc[(pd.isnull(player_list['runs'])), 'runs'] = 0

#how many times did the batsman get out

player_list = player_list.merge(deliveries_2016.groupby('batsman')['player_dismissed'].aggregate({'outs':'count'}), how='left', left_index=True, right_index=True)

player_list['outs'] = 1/player_list['outs']

player_list.loc[(pd.isnull(player_list['outs'])), 'outs'] = 1

player_list.loc[(np.isinf(player_list['outs'])), 'outs'] = 1

#how many times was the batsman involved in a run-out while at the non-striker end

player_list = player_list.merge(deliveries_2016[(deliveries_2016['dismissal_kind']=='run out')].groupby('non_striker')['player_dismissed'].aggregate({'runouts':'count'}), how='left', left_index=True, right_index=True)

player_list['runouts'] = 1/player_list['runouts']

player_list.loc[(pd.isnull(player_list['runouts'])), 'runouts'] = 1

player_list.loc[(np.isinf(player_list['runouts'])), 'runouts'] = 1

#how many 4s did the batsman hit

player_list = player_list.merge(deliveries_2016[(deliveries_2016['batsman_runs']==4)].groupby('batsman')['batsman_runs'].aggregate({'4s':'count'}), how='left', left_index=True, right_index=True)

player_list.loc[(pd.isnull(player_list['4s'])), '4s'] = 0

#how many 6s did the batsman hit

player_list = player_list.merge(deliveries_2016[(deliveries_2016['batsman_runs']==6)].groupby('batsman')['batsman_runs'].aggregate({'6s':'count'}), how='left', left_index=True, right_index=True)

player_list.loc[(pd.isnull(player_list['6s'])), '6s'] = 0

#how many balls did the batsman face

player_list = player_list.merge(deliveries_2016.groupby('batsman')['ball'].aggregate({'balls_faced':'count'}), how='left', left_index=True, right_index=True)

player_list.loc[(pd.isnull(player_list['balls_faced'])), 'balls_faced'] = 0

#how many wickets did the player take

player_list = player_list.merge(deliveries_2016.groupby('bowler')['player_dismissed'].aggregate({'wickets':'count'}), how='left', left_index=True, right_index=True)

player_list.loc[(pd.isnull(player_list['wickets'])), 'wickets'] = 0

#how many runs did the bowler give away

player_list = player_list.merge(deliveries_2016.groupby('bowler')['batsman_runs'].aggregate({'bowl_runs':'sum'}), how='left', left_index=True, right_index=True)

player_list['bowl_runs'] = 1/player_list['bowl_runs']

player_list.loc[(pd.isnull(player_list['bowl_runs'])), 'bowl_runs'] = 1

player_list.loc[(np.isinf(player_list['bowl_runs'])), 'bowl_runs'] = 1

#how many wides did the player bowl

player_list = player_list.merge(deliveries_2016.groupby('bowler')['wide_runs'].aggregate({'wides':'sum'}), how='left', left_index=True, right_index=True)

player_list['wides'] = 1/player_list['wides']

player_list.loc[(pd.isnull(player_list['wides'])), 'wides'] = 1

player_list.loc[(np.isinf(player_list['wides'])), 'wides'] = 1

#how many no-balls did the player bowl

player_list = player_list.merge(deliveries_2016.groupby('bowler')['noball_runs'].aggregate({'noballs':'sum'}), how='left', left_index=True, right_index=True)

player_list['noballs'] = 1/player_list['noballs']

player_list.loc[(pd.isnull(player_list['noballs'])), 'noballs'] = 1

player_list.loc[(np.isinf(player_list['noballs'])), 'noballs'] = 1

#how many balls did the player bowl

player_list = player_list.merge(deliveries_2016.groupby('bowler')['ball'].aggregate({'deliveries':'count'}), how='left', left_index=True, right_index=True)

player_list.loc[(pd.isnull(player_list['deliveries'])), 'deliveries'] = 0

#how many wickets did the player effect as a fielder

player_list = player_list.merge(deliveries_2016.groupby('fielder')['player_dismissed'].aggregate({'wickets_effected':'count'}), how='left', left_index=True, right_index=True)

player_list.loc[(pd.isnull(player_list['wickets_effected'])), 'wickets_effected'] = 0
#get the number of matches each player batted and bowled

player_list['bat_matches'] = deliveries_2016.groupby('batsman')['match_id'].aggregate({'bat_matches':'nunique'})

player_list.loc[(pd.isnull(player_list['bat_matches'])), 'bat_matches'] = 0

player_list['bowl_matches'] = deliveries_2016.groupby('bowler')['match_id'].aggregate({'bowl_matches':'nunique'})

player_list.loc[(pd.isnull(player_list['bowl_matches'])), 'bowl_matches'] = 0
player_list['strike_rate'] = player_list['runs']/player_list['balls_faced']

player_list.loc[(pd.isnull(player_list['strike_rate'])), 'strike_rate'] = 0

player_list['bat_average'] = player_list['runs']/player_list['bat_matches']

player_list.loc[(pd.isnull(player_list['bat_average'])), 'bat_average'] = 0

player_list['4s_per_balls'] = player_list['4s']/player_list['balls_faced']

player_list.loc[(pd.isnull(player_list['4s_per_balls'])), '4s_per_balls'] = 0

player_list['6s_per_balls'] = player_list['6s']/player_list['balls_faced']

player_list.loc[(pd.isnull(player_list['6s_per_balls'])), '6s_per_balls'] = 0

player_list['4s_per_match'] = player_list['4s']/player_list['bat_matches']

player_list.loc[(pd.isnull(player_list['4s_per_match'])), '4s_per_match'] = 0

player_list['6s_per_match'] = player_list['6s']/player_list['bat_matches']

player_list.loc[(pd.isnull(player_list['6s_per_match'])), '6s_per_match'] = 0

player_list['outs_per_match'] = player_list['outs']/player_list['bat_matches']

player_list['outs_per_match'] = 1/player_list['outs_per_match']

player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'outs_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']

player_list.loc[(np.isinf(player_list['outs_per_match'])), 'outs_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']

player_list['runouts_per_match'] = player_list['runouts']/player_list['bat_matches']

player_list['runouts_per_match'] = 1/player_list['runouts_per_match']

player_list.loc[(pd.isnull(player_list['runouts_per_match'])), 'runouts_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']

player_list.loc[(np.isinf(player_list['runouts_per_match'])), 'runouts_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']



player_list['bowl_strike_rate'] = player_list['wickets']/player_list['deliveries']

player_list.loc[(pd.isnull(player_list['bowl_strike_rate'])), 'bowl_strike_rate'] = 0

player_list['bowl_average'] = player_list['bowl_runs']*player_list['bowl_matches'] #since bowl_runs was made reciprocal of itself earlier

player_list.loc[(pd.isnull(player_list['bowl_average'])), 'bowl_average'] = 1

player_list['wides_per_match'] = player_list['wides']*player_list['bowl_matches'] #since wides was made reciprocal of itself earlier

player_list.loc[(pd.isnull(player_list['wides_per_match'])), 'wides_per_match'] = 1

player_list['noballs_per_match'] = player_list['noballs']*player_list['bowl_matches'] #since noballs was made reciprocal of itself earlier

player_list.loc[(pd.isnull(player_list['noballs_per_match'])), 'noballs_per_match'] = 1

player_list['wickets_per_match'] = player_list['wickets']/player_list['bowl_matches']

player_list.loc[(pd.isnull(player_list['wickets_per_match'])), 'wickets_per_match'] = 0
player_list.drop(labels=['balls_faced', 'deliveries', 'bat_matches', 'bowl_matches'], inplace=True, axis=1)
scaler = MinMaxScaler()

player_names = pd.DataFrame(player_list.index)

player_list = scaler.fit_transform(player_list)
num_clusters = len(matches_2016.groupby('team1')['id'].nunique())

kmodel = MiniBatchKMeans(n_clusters=num_clusters, random_state=0).fit(player_list)

player_names['cluster'] = kmodel.predict(player_list)

player_names = pd.DataFrame(player_names.groupby('cluster')['player'].unique())

[print('Cluster ', i+1, ': ', list(player_names.iloc[i,0])) for i in range(num_clusters)]
balanced_teams = []

for i in range(num_clusters):

    player_arr = player_names.iloc[i,0]

    num_sel_per_team = int(np.floor(len(player_arr) / num_clusters))

    rand_indexes = random.sample(range(len(player_arr)), num_sel_per_team*num_clusters)

    for j in range(num_sel_per_team):

        balanced_teams.append(player_arr[rand_indexes[j*8:(j+1)*8]])

balanced_teams = pd.DataFrame(balanced_teams, columns=['Team ' + str(i+1) for i in range(num_clusters)])

print(balanced_teams)