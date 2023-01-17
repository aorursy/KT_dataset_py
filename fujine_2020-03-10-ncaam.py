import os

from pathlib import Path



import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_validate, GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
FILEDIR = Path('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament')
# 提出用ファイルを取得

sub = pd.read_csv(FILEDIR / 'MSampleSubmissionStage1_2020.csv', usecols=['ID'])

id_splited = sub['ID'].str.split('_', expand=True).astype(int).rename(columns={0: 'Season', 1: 'Team1', 2: 'Team2'})

sub = pd.concat([sub, id_splited], axis=1).set_index(['Season', 'Team1', 'Team2']).sort_index()
# シーズン毎の出場チームを抽出

tourney_teams = {}

tourney_teams_all = set()

for season in sub.index.get_level_values('Season').drop_duplicates():

    tourney_teams[season] = set()

    tourney_teams[season].update(sub.loc[season].index.get_level_values('Team1'))

    tourney_teams[season].update(sub.loc[season].index.get_level_values('Team2'))

    tourney_teams_all.update(tourney_teams[season])

{k: len(v) for k, v in tourney_teams.items()}
# 所属カンファレンス情報を取得

conferences = pd.read_csv(FILEDIR / 'MDataFiles_Stage1/MTeamConferences.csv')

conferences = pd.concat(

    [conferences.query('Season == @season and TeamID in @teams') for season, teams in tourney_teams.items()])

conferences = conferences.set_index(['Season', 'TeamID']).sort_index()
# コーチ名を取得

coaches = pd.read_csv(FILEDIR / 'MDataFiles_Stage1/MTeamCoaches.csv')

coaches = pd.concat(

    [coaches.query('Season == @season and TeamID in @team') for season, team in tourney_teams.items()])

coaches = coaches[coaches['LastDayNum'] == 154].set_index(['Season', 'TeamID']).sort_index()[['CoachName']]
# NCAAの初回出場年を取得し、初回出場年から現在までの年数を計算

teams = pd.read_csv(FILEDIR / 'MDataFiles_Stage1/MTeams.csv', usecols=['TeamID', 'FirstD1Season'])

teams['FirstD1Season'] = 2020 - teams['FirstD1Season']

teams = pd.concat(

    [teams.query('TeamID in @team').assign(Season=season) for season, team in tourney_teams.items()])

teams = teams.set_index(['Season', 'TeamID']).sort_index()
# 各シーズンのシードを取得

seeds = pd.read_csv(FILEDIR / 'MDataFiles_Stage1/MNCAATourneySeeds.csv')

seeds = pd.concat(

    [seeds.query('Season == @season and TeamID in @teams') for season, teams in tourney_teams.items()])

seeds = seeds.set_index(['Season', 'TeamID']).sort_index()

seeds['Region'] = seeds['Seed'].str[0]

seeds['Number'] = seeds['Seed'].str[1:3].astype(int)

del seeds['Seed']
# レギュラーシーズンの累計得点と累計失点を取得

regular = pd.read_csv(FILEDIR / 'MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')

regular = regular.drop(columns=['DayNum', 'LTeamID'])

regular = pd.concat(

    [regular.query('Season == @season and WTeamID in @teams') for season, teams in tourney_teams.items()])

regular = regular.groupby(['Season', 'WTeamID']).sum()

regular = regular.rename_axis(index=['Season', 'TeamID'])
# 上記取得データをindexで結合

ctcsr = pd.concat([coaches, teams, conferences, seeds, regular], axis=1)
# NCAAMトーナメントの勝敗結果を取得

result = pd.read_csv(FILEDIR / 'MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

result = result[result['Season'] >= 2015].set_index(['Season', 'WTeamID', 'LTeamID'])
# 各種データと勝敗結果を結合

merged_teams = pd.concat(

    [ctcsr.loc[[(season, wteam), (season, lteam)], :] for season, wteam, lteam, in result.index])



team1 = merged_teams.iloc[::2, :].reset_index('TeamID') # teams winned

team2 = merged_teams.iloc[1::2, :].reset_index('TeamID') # teams losed



merged_teams = pd.concat([

    pd.concat([team1.add_suffix('1'), team2.add_suffix('2')], axis=1).assign(Res=1),

    pd.concat([team2.add_suffix('1'), team1.add_suffix('2')], axis=1).assign(Res=0),

]).reset_index().set_index(['Season', 'TeamID1', 'TeamID2']).sort_index()
# 結合データから、説明変数Xを抽出

# Xを数値化、正規化

x_columns = merged_teams.columns[merged_teams.columns != 'Res']

X = merged_teams[x_columns]



for column in X.select_dtypes(include='number'):

    X[column] = MinMaxScaler().fit_transform(X[column].to_numpy().reshape(-1,1))  # FIXME: SettingWithCopyWarning occurred

X = pd.get_dummies(X, columns=x_columns[X.dtypes == 'object'])
# 目的変数yを設定

y = merged_teams['Res']
# 学習

clfs = {}



# SVC

clfs['SVC'] = {

    'instance': SVC(probability=True),

    'params': [

        {'kernel': ['linear'], 'C': [0.01, 0.05, 0.1, 0.5, 1]},

        {'kernel': ['rbf'], 'C': [1, 10, 50, 100, 250], 'gamma': [0.1, 0.2, 0.3]}

    ]    

}



# RandomForest

clfs['RandomForestClassifier'] = {

    'instance': RandomForestClassifier(n_jobs=-1),

    'params': {        

        'n_estimators': [25, 50, 100],

        'criterion': ['gini', 'entropy'],

        'max_depth': [10, 25, 50, None]

    }

}



# LogisticRegression

clfs['LogisticRegression'] = {

    'instance': LogisticRegression(max_iter=500, n_jobs=-1),

    'params': [

            {'penalty': ['l2'], 'C': [0.1, 0.5, 1, 5, 10]},

            {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': [0.1, 0.5, 1, 5, 10]},

            {'penalty': ['elasticnet'], 'C': [0.1, 0.5, 1, 5, 10], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

        ]

}
for clf_name, clf in clfs.items():

    print('<{}>'.format(clf_name))

    print('  training ...'.format(clf_name))

    

    gs = GridSearchCV(clf['instance'], param_grid=clf['params'], cv=5, n_jobs=-1)

    gs.fit(X, y)

    clfs[clf_name]['best_estimator'] = gs.best_estimator_

    

    print('  best_score: {:.3f}'.format(gs.best_score_))

    print('  best_params: {}'.format(gs.best_params_))
# 各モデルの予測結果から多数決を採用するソフト分類器を作成

vote = VotingClassifier(

    estimators=[(clf_name, clf['best_estimator']) for clf_name, clf in clfs.items()], 

    voting='soft',

    n_jobs=-1

)

vote.fit(X, y)

vote.estimators_
# 各モデルとソフト分類器で、予測結果を比較する

# randomforestだけ精度が異常に高いのは、多分過学習してる。。。

for clf_name, clf in clfs.items():

    score = accuracy_score(y, clf['best_estimator'].predict(X))

    print(clf_name, score)

print('Vote', accuracy_score(y, vote.predict(X)))
# 各モデル個別の予測確率と、全モデルを集約した分類器の予測確率を可視化

# 今回の分類器はソフト投票（各モデルの予測結果の確率の平均をとる）のため、個別のモデルよりもなだらかな分布になっている。

predict_proba = pd.DataFrame(

    {clf_name: clf['best_estimator'].predict_proba(X)[:, 1] for clf_name, clf in clfs.items()},

    index=X.index)

predict_proba['Vote'] = vote.predict_proba(X)[:, 1]

_ = predict_proba.plot(kind='hist', bins=50, grid=True, alpha=0.5, figsize=(16,8))
# 各モデルの予測確率を予測結果ファイルとして出力

columns = predict_proba.columns

for column in columns:

    sub[column] = 0.5



mask = [idx for idx in sub.index if idx in X.index]

sub.loc[mask, columns] = predict_proba.loc[mask, columns]



for column in columns:

    sub[['ID', column]].rename(columns={column: 'pred'}).to_csv('predict_proba-{}.csv'.format(column), index=False)
# 各モデルの勝敗（0か1）を予測結果ファイルとして出力

predict = pd.DataFrame(

    {clf_name: clf['best_estimator'].predict(X) for clf_name, clf in clfs.items()},

    index=X.index)

predict['Vote'] = vote.predict(X)



columns = predict.columns

for column in columns:

    sub[column] = 0.5

    

mask = [idx for idx in sub.index if idx in X.index]

sub.loc[mask, columns] = predict.loc[mask, columns]



for column in columns:

    sub[['ID', column]].rename(columns={column: 'pred'}).to_csv('predict-{}.csv'.format(column), index=False)
import shutil

target_name = 'predict_proba-RandomForestClassifier.csv'

new_name = 'final-submission.csv'

shutil.copy(target_name, new_name)