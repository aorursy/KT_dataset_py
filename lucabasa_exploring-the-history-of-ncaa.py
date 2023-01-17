import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
teams = pd.read_csv('../input/datafiles/Teams.csv')

teams.head()
teams.FirstD1Season.hist(bins=20, alpha=0.7, label='First Season', figsize=(8,5))

teams.LastD1Season.hist(bins=20, alpha=0.7, label='Last Season')

plt.title('Number of teams that joined or left the Division 1', fontsize=15)

plt.xlabel('Year', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.grid(False)

plt.legend()
yr_count = pd.DataFrame({'year': np.arange(1985, 2020)})



for year in yr_count.year:

    teams['is_in'] = 0

    teams.loc[(teams.FirstD1Season <= year) & (teams.LastD1Season >= year), 'is_in'] = 1

    tot_teams = teams.is_in.sum()

    yr_count.loc[yr_count.year == year, 'n_teams'] = tot_teams

    

yr_count = yr_count.set_index('year')

yr_count.n_teams.plot(figsize=(12,6))

plt.title('Number of teams in Division 1', fontsize=15)

plt.xlabel('Year', fontsize=12)

plt.ylabel('N. Teams', fontsize=12)
reg_season = pd.read_csv('../input/datafiles/RegularSeasonCompactResults.csv')

reg_season.head()
reg_season['point_diff'] = reg_season.WScore - reg_season.LScore

reg_season.point_diff.hist(bins=30, figsize=(10,5))

plt.title('Point difference in regular season', fontsize=15)

plt.xlabel('Point difference', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.grid(False)
summaries = reg_season[['Season', 

    'WScore', 

    'LScore', 

    'NumOT', 

    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])



summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]

summaries.sample(10)
fig, ax= plt.subplots(2,2, figsize=(15, 12))



wscores = [col for col in summaries.columns if 'WScore' in col]

lscores = [col for col in summaries.columns if 'LScore' in col]

point_diffs = [col for col in summaries.columns if 'point_diff' in col]



summaries[wscores].plot(ax=ax[0][0], title='Scores of the winning teams', ylim=(15, 190))

ax[0][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])

summaries[lscores].plot(ax=ax[0][1], title='Scores of the losing teams', ylim=(15, 190))

ax[0][1].legend(labels=['Min', 'Max', 'Mean', 'Median'])

summaries[point_diffs].plot(ax=ax[1][0], title='Point differences')

ax[1][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])

summaries[['NumOT_mean']].plot(ax=ax[1][1], title='Average number of OT')

ax[1][1].legend(labels=['Mean'])
summaries = reg_season[['Season', 'WLoc',

    'WScore', 

    'LScore', 

    'NumOT', 

    'point_diff']].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])



summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]

summaries.sample(5)
fig, ax= plt.subplots(3,2, figsize=(15, 18))



wscores = [col for col in summaries.columns if 'WScore' in col]

lscores = [col for col in summaries.columns if 'LScore' in col]

point_diffs = [col for col in summaries.columns if 'point_diff' in col]



summaries[['WScore_mean']].unstack().plot(ax=ax[0][0], title='Avg. scores of the winning teams', ylim=(60,87))

ax[0][0].legend(labels=['Away', 'Home', 'Neutral'])

summaries[['LScore_mean']].unstack().plot(ax=ax[0][1], title='Avg. scores of the losing teams', ylim=(60,87))

ax[0][1].legend(labels=['Away', 'Home', 'Neutral'])

summaries[['point_diff_mean']].unstack().plot(ax=ax[1][0], title='Avg. point differences')

ax[1][0].legend(labels=['Away', 'Home', 'Neutral'])

summaries[['WScore_count']].unstack().plot(ax=ax[1][1], title='Number of wins by location')

ax[1][1].legend(labels=['Away', 'Home', 'Neutral'])

summaries[['NumOT_mean']].unstack().plot(ax=ax[2][0], title='Average number of OT')

ax[2][0].legend(labels=['Away', 'Home', 'Neutral'])

summaries[['NumOT_max']].unstack().plot(ax=ax[2][1], title='Maximum number of OT')

ax[2][1].legend(labels=['Away', 'Home', 'Neutral'])
plt.figure(figsize=(12,10))

sns.scatterplot(reg_season.DayNum, reg_season.point_diff)
def process_details(df):

    data = df.copy()

    stats = [col for col in data.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]



    for col in stats:

        name = col[1:]

        data[name+'_diff'] = data[col] - data['L'+name]

        data[name+'_binary'] = (data[name+'_diff'] > 0).astype(int)

        

    for prefix in ['W', 'L']:

        data[prefix+'FG_perc'] = data[prefix+'FGM'] / data[prefix+'FGA']

        data[prefix+'FGM2'] = data[prefix+'FGM'] - data[prefix+'FGM3']

        data[prefix+'FGA2'] = data[prefix+'FGA'] - data[prefix+'FGA3']

        data[prefix+'FG2_perc'] = data[prefix+'FGM2'] / data[prefix+'FGA2']

        data[prefix+'FG3_perc'] = data[prefix+'FGM3'] / data[prefix+'FGA3']

        data[prefix+'FT_perc'] = data[prefix+'FTM'] / data[prefix+'FTA']

        data[prefix+'Tot_Reb'] = data[prefix+'OR'] + data[prefix+'DR']

        data[prefix+'FGM_no_ast'] = data[prefix+'FGM'] - data[prefix+'Ast']

        data[prefix+'FGM_no_ast_perc'] = data[prefix+'FGM_no_ast'] / data[prefix+'FGM']

        

    data['Game_Rebounds'] = data['WTot_Reb'] + data['LTot_Reb']

    data['WReb_perc'] = data['WTot_Reb'] / data['Game_Rebounds']

    data['LReb_perc'] = data['LTot_Reb'] / data['Game_Rebounds']

    

    return data
reg_season = pd.read_csv('../input/datafiles/RegularSeasonDetailedResults.csv')



stats = [col for col in reg_season.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]



reg_season = process_details(reg_season)



reg_season.head()
not_sum = ['WTeamID', 'DayNum', 'LTeamID']

to_sum = [col for col in reg_season.columns if col not in not_sum]



summaries = reg_season[to_sum].groupby(['Season', 'WLoc']).agg(['min', 'max', 'mean', 'median', 'count'])



summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]

summaries.sample(5)
fig, ax= plt.subplots(7,2, figsize=(15, 6*7))



i, j = 0, 0



for col in stats:

    name = col[1:]

    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].unstack().plot(title='Difference in mean '+name,ax=ax[i][j])

    ax[i][j].legend(labels=['Away', 'Home', 'Neutral'])

    if j == 0: j = 1

    else:

        j = 0

        i += 1
fig, ax= plt.subplots(6,2, figsize=(15, 6*6))



i = 0



for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:

    name = col.split('_perc_')[0][1:]

    summaries[col].unstack().plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])

    summaries['L'+name+'_perc_mean'].unstack().plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])

    ax[i][0].legend(labels=['Away', 'Home', 'Neutral'])

    ax[i][1].legend(labels=['Away', 'Home', 'Neutral'])

    i += 1
playoff = pd.read_csv('../input/datafiles/NCAATourneyCompactResults.csv')

playoff.head()
playoff['point_diff'] = playoff.WScore - playoff.LScore

playoff.point_diff.hist(bins=30, figsize=(10,5))

plt.title('Point difference in the playoffs', fontsize=15)

plt.xlabel('Point difference', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.grid(False)
summaries = playoff[['Season', 

    'WScore', 

    'LScore', 

    'NumOT', 

    'point_diff']].groupby('Season').agg(['min', 'max', 'mean', 'median'])



summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]



fig, ax= plt.subplots(2,2, figsize=(15, 12))



wscores = [col for col in summaries.columns if 'WScore' in col]

lscores = [col for col in summaries.columns if 'LScore' in col]

point_diffs = [col for col in summaries.columns if 'point_diff' in col]



summaries[wscores].plot(ax=ax[0][0], title='Scores of the winning teams', ylim=(25, 160))

ax[0][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])

summaries[lscores].plot(ax=ax[0][1], title='Scores of the losing teams', ylim=(25, 160))

ax[0][1].legend(labels=['Min', 'Max', 'Mean', 'Median'])

summaries[point_diffs].plot(ax=ax[1][0], title='Point differences')

ax[1][0].legend(labels=['Min', 'Max', 'Mean', 'Median'])

summaries[['NumOT_mean']].plot(ax=ax[1][1], title='Average number of OT')

ax[1][1].legend(labels=['Mean'])
playoff = pd.read_csv('../input/datafiles/NCAATourneyDetailedResults.csv')



stats = [col for col in playoff.columns if 'W' in col and 'ID' not in col and 'Loc' not in col]



playoff= process_details(playoff)



not_sum = ['WTeamID', 'DayNum', 'LTeamID']

to_sum = [col for col in reg_season.columns if col not in not_sum]



summaries = playoff[to_sum].groupby(['Season']).agg(['min', 'max', 'mean', 'median', 'count'])



summaries.columns = ['_'.join(col).strip() for col in summaries.columns.values]



fig, ax= plt.subplots(7,2, figsize=(15, 6*7))



i, j = 0, 0



for col in stats:

    name = col[1:]

    summaries[[c for c in summaries.columns if name+'_diff_mean' in c]].plot(title='Difference in mean '+name,ax=ax[i][j])

    if j == 0: j = 1

    else:

        j = 0

        i += 1
fig, ax= plt.subplots(6,2, figsize=(15, 6*6))



i = 0



for col in [c for c in summaries.columns if '_perc_mean' in c and c.startswith('W')]:

    name = col.split('_perc_')[0][1:]

    summaries[col].plot(title='Mean percenteage of '+name+', Winners',ax=ax[i][0])

    summaries['L'+name+'_perc_mean'].plot(title='Mean percenteage of '+name+', Losers',ax=ax[i][1])

    i += 1