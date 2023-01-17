import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

import numpy as np





# plotly imports

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=True)



import shap



# load JS visualization code to notebook

shap.initjs()





base_address = '../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data'



m_teams = pd.read_csv(base_address + '/MDataFiles_Stage1/MTeams.csv')



# 2019 all tournament compat results, meaning each game score

# this file will be useful for getting overall birds eye view

tourn_compat_results = pd.read_csv(base_address + '/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

tourn_compat_results_2019 = tourn_compat_results.loc[tourn_compat_results.Season==2019]



tourn_detail_results = pd.read_csv(base_address + '/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')

tourn_detail_results_2019 = tourn_detail_results.loc[tourn_detail_results.Season==2019]





tourney_seeds = pd.read_csv(base_address + '/MDataFiles_Stage1/MNCAATourneySeeds.csv')
############

# this cell process dataset, merge and calculate upsets game

###########



tourn_detail_results_merge = tourn_detail_results.merge(tourney_seeds, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='inner')

tourn_detail_results_merge = tourn_detail_results_merge.loc[tourn_detail_results_merge.DayNum.isin([136, 137])]

tourn_detail_results_merge = tourn_detail_results_merge.rename(columns={'Seed': 'WSeed'})

tourn_detail_results_merge = tourn_detail_results_merge.drop(['TeamID'], 1)



tourn_detail_results_merge = tourn_detail_results_merge.merge(tourney_seeds, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='inner')

tourn_detail_results_merge = tourn_detail_results_merge.loc[tourn_detail_results_merge.DayNum.isin([136, 137])]

tourn_detail_results_merge = tourn_detail_results_merge.rename(columns={'Seed': 'LSeed'})

tourn_detail_results_merge = tourn_detail_results_merge.drop(['TeamID'], 1)







tourn_detail_results_merge = tourn_detail_results_merge.merge(m_teams[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID',how='inner')

tourn_detail_results_merge = tourn_detail_results_merge.rename(columns={'TeamName': 'WTeamName'})

tourn_detail_results_merge = tourn_detail_results_merge.drop(['TeamID'], 1)



tourn_detail_results_merge = tourn_detail_results_merge.merge(m_teams[['TeamID', 'TeamName']], left_on='LTeamID', right_on='TeamID',how='inner')

tourn_detail_results_merge = tourn_detail_results_merge.rename(columns={'TeamName': 'LTeamName'})

tourn_detail_results_merge = tourn_detail_results_merge.drop(['TeamID'], 1)











tourn_detail_results_merge['WSeedNum'] = tourn_detail_results_merge.WSeed.str.extract('(\d+)')

tourn_detail_results_merge['LSeedNum'] = tourn_detail_results_merge.LSeed.str.extract('(\d+)')

tourn_detail_results_merge['WSeedNum'] = tourn_detail_results_merge['WSeedNum'].astype(int)

tourn_detail_results_merge['LSeedNum'] = tourn_detail_results_merge['LSeedNum'].astype(int)





tourn_detail_results_merge_upset = tourn_detail_results_merge.loc[tourn_detail_results_merge.WSeedNum > 10]
###############

# this cell caluclate adavance features and also it 

# seperate dataframe for loser and winners

###############



winner = tourn_detail_results_merge_upset[['WFGM', 'WFGA', 'WFGM3',

                                           'WFGA3', 'WFTM', 'WFTA',

                                           'WOR', 'WDR','WAst',

                                           'WTO', 'WStl',

                                           'WBlk', 'WPF', 'LDR', 'LOR']]

winner.columns = ['FGM', 'FGA', 'FGM3','FGA3', 'FTM', 'FTA',

                  'OR', 'DR','Ast','TO', 'Stl','Blk', 'PF', 'LDR', 'LOR']

winner['status'] = 'Winner'

winner['FGP'] = winner['FGM'] / winner['FGA']

winner['FGP2'] = (winner['FGM'] - winner['FGM3']) / (winner['FGA'] - winner['FGA3'])

winner['FGP3'] = winner['FGM3'] / winner['FGA3']

winner['FTP'] = winner['FTM'] / winner['FTA']

winner['ORP'] = winner['OR'] / (winner['OR']+winner['LDR'])

winner['DRP'] = winner['DR'] / (winner['DR']+winner['LOR'])

winner['POS'] = 0.96 * (winner['FGA'] + winner['TO'] + 0.44 * winner['FTA'] - winner['OR'])













looser = tourn_detail_results_merge_upset[['LFGM', 'LFGA', 'LFGM3', 'LFGA3',

       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WDR', 'WOR']]



looser.columns = ['FGM', 'FGA', 'FGM3', 'FGA3',

       'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'WDR', 'WOR']



looser['status'] = 'Looser'

looser['FGP'] = looser['FGM'] / looser['FGA']

looser['FGP2'] = (looser['FGM'] - looser['FGM3']) / (looser['FGA'] - looser['FGA3'])

looser['FGP3'] = looser['FGM3'] / looser['FGA3']

looser['FTP'] = looser['FTM'] / looser['FTA']

looser['ORP'] = looser['OR'] / (looser['OR']+looser['WDR'])

looser['DRP'] = looser['DR'] / (looser['DR']+looser['WOR'])

looser['POS'] = 0.96 * (looser['FGA'] + looser['TO'] + 0.44 * looser['FTA'] - looser['OR'])







basic_stats_features = ['FGM', 'FGA', 'FGM3', 'FGA3',

       'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']

advanced_stats_features = ['status', 'FGP', 'FGP2', 'FGP3',

                           'FTP', 'ORP', 'DRP', 'POS']





df_stats = pd.concat([winner[basic_stats_features+advanced_stats_features],

           looser[basic_stats_features+advanced_stats_features]])
temp = df_stats[basic_stats_features+ ['status']].groupby('status').mean()

temp = temp.rename_axis(None)

temp.style.background_gradient(cmap='Blues')
temp = df_stats[advanced_stats_features].groupby('status').mean()

temp = temp.rename_axis(None)

temp.style.background_gradient(cmap='Blues')
mens_events = []

for year in [2015, 2016, 2017, 2018, 2019]:

    mens_events.append(pd.read_csv(f'../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MEvents{year}.csv'))

MEvents = pd.concat(mens_events)







MEvents_upset = MEvents.merge(tourn_detail_results_merge_upset[['Season', 'DayNum', 'WTeamID', 'WScore',

                                                'LTeamID','WSeed', 'LSeed', 'WTeamName', 

                                                'LTeamName', 'WSeedNum', 'LSeedNum']],

             on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='inner')



MEvents_upset.loc[MEvents_upset.EventTeamID == MEvents_upset.WTeamID, 'event_team_status'] = 'Winner'

MEvents_upset['event_team_status'] = MEvents_upset['event_team_status'].fillna('Looser')
temp = MEvents_upset.groupby(['Season', 'WTeamID', 'LTeamID','event_team_status'])['EventType'].value_counts().reset_index(name='count')

temp2 = temp.groupby(['event_team_status', 'EventType'])['count'].mean().reset_index()

fig = px.bar(temp2, x='EventType', y='count', color='event_team_status', barmode='group')

for axis in fig.layout:

    if type(fig.layout[axis]) == go.layout.YAxis:

        fig.layout[axis].title.text = ''

fig.show()
MEvents_upset.loc[MEvents_upset.ElapsedSeconds <= 1200, 'time_quadrent'] = 'first quarter'

MEvents_upset.loc[(MEvents_upset.ElapsedSeconds > 1200) & (MEvents_upset.ElapsedSeconds <=2400), 'time_quadrent'] = 'second quarter'

MEvents_upset.loc[MEvents_upset.ElapsedSeconds > 2400, 'time_quadrent'] = 'extra'









temp = MEvents_upset.groupby(['Season', 'WTeamID', 'LTeamID','event_team_status', 'time_quadrent'])['EventType'].value_counts().reset_index(name='count')

temp2 = temp.groupby(['event_team_status', 'EventType', 'time_quadrent'])['count'].mean().reset_index()

fig = px.bar(temp2, x='EventType', y='count', color='event_team_status', barmode='group', facet_row='time_quadrent')



for axis in fig.layout:

    if type(fig.layout[axis]) == go.layout.YAxis:

        fig.layout[axis].title.text = ''



for annotation in fig.layout.annotations:

    annotation.text = annotation.text.split("=")[1]

    

fig.show()
temp = (MEvents_upset.loc[MEvents_upset.Season > 2018]).sort_values(['Season', 'WTeamID', 'LTeamID','ElapsedSeconds'])

temp['score_diff'] = temp['WCurrentScore'] - temp['LCurrentScore']



fig = px.line(temp, x="ElapsedSeconds", y="score_diff", facet_row='LSeed')

fig.show()
##########

# this cell train a randomtreeclassifier and used for model explainability

#########



tourn_detail_results_merge_regular = tourn_detail_results_merge.loc[tourn_detail_results_merge.WSeedNum < 5]

tourn_detail_results_merge_regular = tourn_detail_results_merge_regular.head(83)



tourn_detail_results_merge_regular['winning_term'] = '0'



tourn_detail_results_merge_upset['winning_term'] = '1'









train_col = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',

       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',

       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']





train = pd.concat([tourn_detail_results_merge_upset, tourn_detail_results_merge_regular])

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier



X = train[train_col]

Y = train['winning_term']

clf = RandomForestClassifier(random_state=0)

clf = clf.fit(X, Y)
explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X.iloc[4])



shap.force_plot(explainer.expected_value[1], shap_values[1], X.iloc[4])