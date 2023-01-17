# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')



import matplotlib.pyplot as plt

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import scipy.stats as stats

from scipy.interpolate import interp1d

import statsmodels.api as sm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# %matplotlib inline

# Any results you write to the current directory are saved as output.
# team

teams = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv')

# season

seasons = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MSeasons.csv')

# seed

seeds = pd.read_csv(r'/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneySeeds.csv')

# finalists

finalists = pd.read_csv(r'/kaggle/input/ncaa-d1-finalists-ever/finalists.csv')

# season compact results

MRegularSeasonCompactResults = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')

# tourney compact results

MNCAATourneyCompactResults = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')

# coaches

coaches = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeamCoaches.csv')

# events

events2015 = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2015.csv')

events2016 = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2016.csv')

events2017 = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2017.csv')

events2018 = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2018.csv')

events2019 = pd.read_csv('/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents2019.csv')
teams['FirstD1Season'].plot.hist(legend=True), teams['LastD1Season'].plot.hist(legend=True)
team_amounts = []



for i in range(1985, 2020):

    team_amounts.append(len(np.unique(MRegularSeasonCompactResults.groupby('Season').get_group(i)[['WTeamID', 'LTeamID']].values)))



plt.plot(seasons['Season'][:-1], team_amounts)

plt.title(label='Number of Teams per Regular Season')
plt.figure(figsize=(14, 10))



regions = {'RegionW': 'r', 'RegionX': 'g', 'RegionY': 'b', 'RegionZ': 'y'}



for i, key in enumerate(regions):

    plt.subplot(2, 2, i+1)

    title = key[:-1] + ' ' + key[-1]

    seasons[key].drop(axis=0, index=35).value_counts().sort_values().plot.barh(title=title, color=regions[key])
# look up the rank of each team in each season

rank = seeds.merge(teams)

rank['Rank'] = rank['Seed'].apply(lambda x : re.findall(r'([0-9]*)', x))



for i in range(len(rank['Rank'])):

    a = rank['Rank'].iloc[i]

    b = int(a[1])

    rank['Rank'].iloc[i] = b
rank[['Season', 'TeamName', 'Rank']]
plt.figure(figsize=(20, 20))



for i in range(16):

    plt.subplot(4, 4, 1+i)

    seed = rank.groupby(['Rank']).get_group(1 + i)

    first4_seed = seed['TeamName'].value_counts().sort_values(ascending=False).head(4)

    if i == 0:

        first4_seed.plot.barh(title=str(i+1) + 'st seed', color='#00' + str(8000 + 1000 * i))

    elif i == 1:

        first4_seed.plot.barh(title=str(i+1) + 'nd seed', color='#00' + str(8000 + 1000 * i))

    elif i == 2:

        first4_seed.plot.barh(title=str(i+1) + 'rd seed', color='#0' + str(8000 + 1000 * i))

    else:

        first4_seed.plot.barh(title=str(i+1) + 'th seed', color='#0' + str(8000 + 1000 * i))
fig = plt.figure(figsize=(12, 12))



colleges = {'North Carolina': 'c', 'Duke': 'm', 'Kentucky': 'y', 'Kansas': 'k'}



for i, key in enumerate(colleges):

    ax = fig.add_subplot(411+i)

    college = rank.groupby(['TeamName']).get_group(key)

    ax.plot(college['Season'], college['Rank'], label=key, color=colleges[key], marker='o')

    plt.gca().invert_yaxis()

    ax.legend()
# only consider the finalists since 1985

finalists = finalists.head(35)



top_seeds = pd.merge(pd.merge(finalists, rank, left_on=['champion', 'year'], right_on=['TeamName', 'Season']),rank, 

         left_on=['runner_up', 'year'], right_on=['TeamName', 'Season'], suffixes=('_champion', '_runner_up'))



top_seeds.plot.bar(x='year', y=['Rank_champion', 'Rank_runner_up'], legend='reverse')
MRegularSeasonCompactResults['WLoc'].value_counts().plot.bar()
ax = sns.distplot(np.log(MRegularSeasonCompactResults['WScore']), fit_kws={"color":"red"}, kde=False,

        fit=stats.gamma, hist=None, label="Winners' score distribution(Log Scale)")

ax = sns.distplot(np.log(MRegularSeasonCompactResults['LScore']), fit_kws={"color":"blue"}, kde=False,

        fit=stats.gamma, hist=None, label="Losers' score distribution(Log Scale)")

ax.legend()

l1 = ax.lines[0]

l2 = ax.lines[1]

x1 = l1.get_xydata()[:,0]

y1 = l1.get_xydata()[:,1]

x2 = l2.get_xydata()[:,0]

y2 = l2.get_xydata()[:,1]

ax.fill_between(x1,y1, color="red", alpha=0.3)

ax.fill_between(x2,y2, color="blue", alpha=0.3)

print('A winner on average scores {} more points than its opponent.'.format(math.exp(y1.mean())-math.exp(y2.mean())))

plt.show(block=False)
MRegularSeasonCompactResults['Score_Difference'] = np.log(MRegularSeasonCompactResults['WScore'] - MRegularSeasonCompactResults['LScore'])

ax = sns.distplot(MRegularSeasonCompactResults['Score_Difference'], fit_kws={"color":"green"}, kde=False,

        fit=stats.gamma, hist=None, label="Score Difference Distribution (Log Scale)")



ax.legend()



l = ax.lines[0]



x = l.get_xydata()[:,0]

y = l.get_xydata()[:,1]



ax.fill_between(x,y, color="green", alpha=0.3)

print('The average gap of a game is  {}.'.format(math.exp(y.mean())))

plt.show(block=False)
pd.merge(MRegularSeasonCompactResults, teams, left_on='WTeamID', right_on='TeamID')['TeamName'].value_counts().head(5).sort_values().plot.barh(title='Most winning teams(top 5)')
pd.merge(MRegularSeasonCompactResults, teams, left_on='WTeamID', right_on='TeamID').groupby('TeamName')['NumOT'].sum().sort_values(ascending=False).head(5).sort_values().plot.barh(title='Top 5 teams winning most OH')
# classify the stages

MNCAATourneyCompactResults['Stage'] = ''

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 134] = 'First Four'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 135] = 'First Four'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 136] = 'First Round'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 137] = 'First Round'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 138] = 'Second Round'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 139] = 'Second Round'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 143] = 'Regional Semifinals'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 144] = 'Regional Semifinals'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 145] = 'Regional Finals'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 146] = 'Regional Finals'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 152] = 'National Semifinals'

MNCAATourneyCompactResults['Stage'].loc[MNCAATourneyCompactResults['DayNum'] == 154] = 'National Finals'
game_records = pd.merge(pd.merge(MNCAATourneyCompactResults, teams[['TeamID','TeamName']], left_on='WTeamID', right_on='TeamID'), teams[['TeamID','TeamName']], left_on='LTeamID', right_on='TeamID', suffixes=('_W','_L'))
game_records['TeamName_W'].loc[game_records['Stage'] == 'National Finals'].value_counts().head(5).sort_values(ascending=True).plot.barh()
colleges = {'Duke': [], 'North Carolina': [], 'Kansas': [], 'Kentucky': []}



for name, _list in colleges.items():

    for i in range(1985, 2020):

        try:

            result = game_records.groupby('Season').get_group(i).groupby('TeamName_L').get_group(name)['Stage']

            _list.append(result.iloc[0])

        except:

            if name in np.unique(game_records.groupby('Season').get_group(i)['TeamName_W']):

                _list.append('National Championship')

            else:

                _list.append('N/A')
rounds = ['N/A', 'First Four', 'First Round', 'Second Round', 'Regional Semifinals', 'Regional Finals', 

          'National Semifinals', 'National Finals', 'National Championship']

y_pos = np.arange(float(len(rounds)))

fig = plt.figure(figsize=(20, 12))



for i, college in enumerate(['Duke', 'North Carolina', 'Kentucky', 'Kansas']):

    ax = fig.add_subplot(221+i)

    plt.yticks(y_pos, rounds)



    y = pd.Series(colleges[college]).apply(lambda x: ['N/A', 'First Four', 'First Round', 'Second Round', 'Regional Semifinals', 'Regional Finals', 

              'National Semifinals', 'National Finals', 'National Championship'].index(x))

    x = np.squeeze(pd.DataFrame(np.arange(1985.0, 2020.0)).values)



    lowess = sm.nonparametric.lowess(y, x, frac=.3)

    lowess_x = list(zip(*lowess))[0]

    lowess_y = list(zip(*lowess))[1]



    f = interp1d(lowess_x, lowess_y, bounds_error=False)



    xnew = [i/1. for i in range(1985, 2020)]

    ynew = f(xnew)



    plt.plot(x, y, '-o')

    plt.title(college)

    plt.plot(xnew, ynew, '-')
new_game_records = pd.merge(pd.merge(MRegularSeasonCompactResults, rank[['Season', 'TeamID', 'Rank', 'TeamName']], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']), rank[['Season', 'TeamID', 'Rank', 'TeamName']], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))
totals = []

# calculate win percentage

for i in range(1985, 2020):

    winDict = {k+1: 0 for k in range(16)}

    lossDict = {k+1: 0 for k in range(16)}

    wins = new_game_records.groupby('Season').get_group(i)['Rank_W'].value_counts().to_dict()

    winDict.update(wins)

    losses = new_game_records.groupby('Season').get_group(i)['Rank_L'].value_counts().to_dict()

    lossDict.update(losses)

    totalDict = {key: winDict[key]/(winDict[key]+lossDict[key]) for key in winDict.keys()}

    totals.append(totalDict)
x = []

y = []

for t in totals:

    x.extend(list(t.keys()))

    y.extend(list(t.values()))

    

fig, ax = plt.subplots()

ax.set_xlim(17,0)



lowess = sm.nonparametric.lowess(y, x, frac=.3)

lowess_x = list(zip(*lowess))[0]

lowess_y = list(zip(*lowess))[1]



f = interp1d(lowess_x, lowess_y, bounds_error=False)



xnew = [i/1. for i in range(16*35)]

ynew = f(xnew)



plt.scatter(x,y,color='c')

plt.plot(xnew, ynew, '-',color='g')

plt.show()
coach_games = pd.merge(pd.merge(game_records[['Season','WTeamID', 'LTeamID']], coaches, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']), 

        coaches, left_on=['Season','LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_W', '_L'))
wins = dict() # collect the wins of coaches

totals = dict() # colect games played of coaches



for i in range(len(coach_games)):

    if coach_games['CoachName_W'].iloc[i] in wins:

        wins[coach_games['CoachName_W'].iloc[i]] += 1

        totals[coach_games['CoachName_W'].iloc[i]] += 1

    else:

        wins[coach_games['CoachName_W'].iloc[i]] = 1

        totals[coach_games['CoachName_W'].iloc[i]] = 1

        

    if coach_games['CoachName_L'].iloc[i] in totals:

        totals[coach_games['CoachName_L'].iloc[i]] += 1

    else:

        totals[coach_games['CoachName_L'].iloc[i]] = 1
labels = 'mike_krzyzewski', 'roy_williams', 'john_calipari', 'jim_boeheim', 'rick_pitino', 'others'

sizes = [wins['mike_krzyzewski']/sum(wins.values()), wins['roy_williams']/sum(wins.values()), wins['john_calipari']/sum(wins.values()),

         wins['jim_boeheim']/sum(wins.values()), wins['rick_pitino']/sum(wins.values()), 

         1-(wins['mike_krzyzewski']+wins['roy_williams']+wins['john_calipari']+wins['jim_boeheim']+wins['rick_pitino'])/sum(wins.values())]

explode = (0.5, 0.0, 0, 0,0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
rates = dict()



for k1, v1 in wins.items():

    for k2, v2 in totals.items():

        if v2 >= 50 and k1 == k2:

            rates[k1] = v1/v2
rates = {k: v for k, v in sorted(rates.items(), key=lambda item: item[1], reverse=True)}



print("The winning percentage of the most successful coaches:")

for k,v in rates.items():

    print(k, ":", v)
team_stat_2015 = dict()

team_stat_2016 = dict()

team_stat_2017 = dict()

team_stat_2018 = dict()

team_stat_2019 = dict()
def archive_team_stat(events, team_stat):

    for team in events['EventTeamID'].unique():

        if team not in team_stat:

            team_stat[team] = dict()



        team_stat[team]['matches'] = list()

        team_stat[team]['field_goals'] = 0

        team_stat[team]['field_goals_attempted'] = 0

        team_stat[team]['three_pointers'] = 0

        team_stat[team]['three_pointers_attempted'] = 0



        info = events.groupby('EventTeamID').get_group(team)



        for i in range(info.shape[0]):

            if info['DayNum'].iloc[i] not in team_stat[team]['matches']:

                team_stat[team]['matches'].append(info['DayNum'].iloc[i])

            elif info['EventType'].iloc[i] == 'made2':

                team_stat[team]['field_goals'] += 1

                team_stat[team]['field_goals_attempted'] += 1

            elif info['EventType'].iloc[i] == 'miss2':

                team_stat[team]['field_goals_attempted'] += 1

            elif info['EventType'].iloc[i] == 'made3':

                team_stat[team]['three_pointers'] += 1

                team_stat[team]['three_pointers_attempted'] += 1

            elif info['EventType'].iloc[i] == 'miss3':

                team_stat[team]['three_pointers_attempted'] += 1

                

    for team, stat in team_stat.items():

        try:

            team_stat[team]['EFG'] = team_stat[team]['field_goals'] + team_stat[team]['three_pointers'] + 0.5 * team_stat[team]['three_pointers'] / (team_stat[team]['field_goals_attempted'] + team_stat[team]['three_pointers_attempted'])

            team_stat[team]['EFG'] = team_stat[team]['EFG'] / len(team_stat[team]['matches'])

        except:

            team_stat[team]['EFG'] = 0

    return team_stat
team_stats = [team_stat_2015, team_stat_2016, team_stat_2017, team_stat_2018, team_stat_2019]

events = [events2015, events2016, events2017, events2018, events2019]



for i in range(len(team_stats)):

    team_stats[i] = archive_team_stat(events[i], team_stats[i])

    # sort by the value of EFG

    team_stats[i] = {k: v for k, v in sorted(team_stats[i].items(), key=lambda item: item[1]['EFG'], reverse=True)}

    team_stats[i] = {k: v for k, v in team_stats[i].items() if  k not in np.unique(game_records.groupby('Season').get_group(i+2015)[['TeamID_W', 'TeamID_L']])}
efficiency_vs_performance_2015 = pd.DataFrame(columns=['TeamID', 'TeamName', 'EFG', 'Stage'])

efficiency_vs_performance_2016 = pd.DataFrame(columns=['TeamID', 'TeamName', 'EFG', 'Stage'])

efficiency_vs_performance_2017 = pd.DataFrame(columns=['TeamID', 'TeamName', 'EFG', 'Stage'])

efficiency_vs_performance_2018 = pd.DataFrame(columns=['TeamID', 'TeamName', 'EFG', 'Stage'])

efficiency_vs_performance_2019 = pd.DataFrame(columns=['TeamID', 'TeamName', 'EFG', 'Stage'])



evsp = [efficiency_vs_performance_2015, efficiency_vs_performance_2016, efficiency_vs_performance_2017, efficiency_vs_performance_2018, efficiency_vs_performance_2019]
def efficiency_archive(index, df, team_stat):

    for team in np.unique(game_records.groupby('Season').get_group(index+2015)[['TeamID_W', 'TeamID_L']]):

        try:

            team_name = game_records.groupby(['Season', 'TeamID_W']).get_group((index+2015, team))['TeamName_W'].iloc[0]

        except:

            team_name = game_records.groupby(['Season', 'TeamID_L']).get_group((index+2015, team))['TeamName_L'].iloc[0]

#         print(team_name)

        try:

            win_day = game_records.groupby(['Season', 'TeamID_W']).get_group((index+2015, team))['DayNum'].max()

        except:

            win_day = 0

        try:

            loss_day = game_records.groupby(['Season', 'TeamID_L']).get_group((index+2015, team))['DayNum'].max()

        except:

            loss_day = 0

#         print(max(loss_day, win_day))

        if max(loss_day, win_day) == 134:

            stage = 'First Four'

        elif max(loss_day, win_day) == 135:

            stage = 'First Four'

        elif max(loss_day, win_day) == 136:

            stage = 'First Round'  

        elif max(loss_day, win_day) == 137:

            stage = 'First Round'

        elif max(loss_day, win_day) == 138:

            stage = 'Second Round'

        elif max(loss_day, win_day) == 139:

            stage = 'Second Round'

        elif max(loss_day, win_day) == 143:

            stage = 'Regional Semifinals'

        elif max(loss_day, win_day) == 144:

            stage = 'Regional Semifinals'

        elif max(loss_day, win_day) == 145:

            stage = 'Regional Finals'

        elif max(loss_day, win_day) == 146:

            stage = 'Regional Finals'

        elif max(loss_day, win_day) == 152:

            stage = 'National Semifinals'

        elif max(loss_day, win_day) == 154:

            if loss_day == 154:

                stage = 'National Finals'

            else:

                stage = 'National Championship'

        try:

            new_row = pd.DataFrame({

                          'TeamID': [team],

                          'TeamName': [team_name],

                          'EFG': [team_stat[team]['EFG']],

                          'Stage': [stage]

                      })

        except:

            if index == 0:

                new_row = pd.DataFrame({

                              'TeamID': [team],

                              'TeamName': [team_name],

                              'EFG': [team_stat_2015[team]['EFG']],

                              'Stage': [stage]

                          })

            elif index == 1:

                new_row = pd.DataFrame({

                              'TeamID': [team],

                              'TeamName': [team_name],

                              'EFG': [team_stat_2016[team]['EFG']],

                              'Stage': [stage]

                          })

            elif index == 2:

                new_row = pd.DataFrame({

                              'TeamID': [team],

                              'TeamName': [team_name],

                              'EFG': [team_stat_2017[team]['EFG']],

                              'Stage': [stage]

                          })

            elif index == 3:

                new_row = pd.DataFrame({

                              'TeamID': [team],

                              'TeamName': [team_name],

                              'EFG': [team_stat_2018[team]['EFG']],

                              'Stage': [stage]

                          })

            elif index == 4:

                new_row = pd.DataFrame({

                              'TeamID': [team],

                              'TeamName': [team_name],

                              'EFG': [team_stat_2019[team]['EFG']],

                              'Stage': [stage]

                          })

    #     print(new_row)

        df = pd.concat([df, new_row], ignore_index=True)

        

    return df
for i in range(len(evsp)):

    evsp[i] = efficiency_archive(i, evsp[i], team_stats[i])
rounds = ['First Four', 'First Round', 'Second Round', 'Regional Semifinals', 'Regional Finals', 

          'National Semifinals', 'National Finals', 'National Championship']

y_pos = np.arange(float(len(rounds)))

fig = plt.figure(figsize=(15, 80))



for i in range(len(evsp)):

    ax = fig.add_subplot(511+i)

    y = evsp[i]['Stage'].apply(lambda x: ['First Four', 'First Round', 'Second Round', 'Regional Semifinals', 'Regional Finals', 

          'National Semifinals', 'National Finals', 'National Championship'].index(x))

    x = evsp[i]['EFG']



    chart = sns.regplot(x, y, lowess=True)

    labels = [item.get_text() for item in chart.get_yticklabels()]

    for j in range(len(rounds)):

        labels[1+j] = rounds[j]

    chart.set_yticklabels(labels)

    plt.title(str(2015+i))
def calculate_final_EFG(df, event):

    df.insert(2, "Final_EFG", [0 for _ in range(df.shape[0])], True) 

    for team in event['EventTeamID'].unique():

        if team != 0:

            field_goals = 0

            field_goals_attempted = 0

            three_pointers = 0

            three_pointers_attempted = 0



            try:

                last_loss = max(event.groupby('LTeamID').get_group(team)['DayNum'])

            except:

                # a team did not lose

                last_loss = 0



            try:

                last_win = max(event.groupby('WTeamID').get_group(team)['DayNum'])

            except:

                # a team did not win

                last_win = 0



            final_match = max(last_loss, last_win)

            try:

                info = event.groupby(['EventTeamID', 'DayNum']).get_group((team, final_match))

                for i in range(info.shape[0]):

                    if info['EventType'].iloc[i] == 'made2':

                        field_goals += 1

                        field_goals_attempted += 1

                    elif info['EventType'].iloc[i] == 'miss2':

                        field_goals_attempted += 1

                    elif info['EventType'].iloc[i] == 'made3':

                        three_pointers += 1

                        three_pointers_attempted += 1

                    elif info['EventType'].iloc[i] == 'miss3':

                        three_pointers_attempted += 1



                new_EFG = 0

                new_EFG = field_goals + three_pointers + 0.5 * three_pointers / (field_goals_attempted + three_pointers_attempted)

                df['Final_EFG'].loc[df['TeamID'] == team] = new_EFG

            except:

                print('ERROR!')

                

    return df
for i in range(len(evsp)):

    evsp[i] = calculate_final_EFG(evsp[i], events[i])
losers = pd.DataFrame(columns=['TeamID', 'TeamName', 'Final_EFG', 'EFG', 'Stage'])



for i in range(len(evsp)):

    # eliminate dirty data

    l = evsp[i].drop(evsp[i][evsp[i]['Final_EFG'] == 0].index)

    # eliminate champions' data

    if i == 0:

        l = l.drop(l[l['TeamName'] == 'Duke'].index)

    elif i == 1:

        l = l.drop(l[l['TeamName'] == 'Villanova'].index)

    elif i == 2:

        l = l.drop(l[l['TeamName'] == 'North Carolina'].index)

    elif i == 3:

        l = l.drop(l[l['TeamName'] == 'Villanova'].index)

    elif i == 4:

        l = l.drop(l[l['TeamName'] == 'Virginia'].index)

        

    losers = pd.concat([losers, l], ignore_index=True)
losers['EFG_Difference'] = losers['EFG'] - losers['Final_EFG']

ax = sns.distplot(losers['EFG_Difference'], fit_kws={"color":"green"}, kde=False,

        fit=stats.gamma, hist=None, label="EFG Difference")



ax.legend()



l = ax.lines[0]



x = l.get_xydata()[:,0]

y = l.get_xydata()[:,1]



ax.fill_between(x,y, color="green", alpha=0.3)

print('The average difference of EFG from average performance is  {}.'.format(math.exp(y.mean())))

plt.show(block=False)
champions = pd.DataFrame(columns=['TeamID', 'TeamName', 'Final_EFG', 'EFG', 'Stage'])



champions = pd.concat([champions, evsp[0][evsp[0]['TeamName'] == 'Duke']])

champions = pd.concat([champions, evsp[1][evsp[1]['TeamName'] == 'Villanova']])

champions = pd.concat([champions, evsp[2][evsp[2]['TeamName'] == 'North Carolina']])

champions = pd.concat([champions, evsp[3][evsp[3]['TeamName'] == 'Villanova']])

champions = pd.concat([champions, evsp[4][evsp[4]['TeamName'] == 'Virginia']])
ind = np.arange(5)

width = 0.35       

plt.bar(ind, champions['Final_EFG'], width, label='EFG in Final')

plt.bar(ind + width, champions['EFG'], width,

    label='Average EFG')



plt.ylabel('EFG')

plt.title('EFG of National Champions (2015-2019)')



plt.xticks(ind + width / 2, ('2015\nDuke', '2016\nVillanova', '2017\nNorth Carolina', '2018\nVillanova', '2019\nVirginia'))

plt.legend(loc='best')

plt.show()