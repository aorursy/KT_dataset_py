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
# 結合データから目的変数（Res）の列を除き、説明変数Xを抽出

x_columns = merged_teams.columns[merged_teams.columns != 'Res']

X = merged_teams[x_columns]



# 数値の列を正規化し、文字の列をダミー変数に変換

columns_number = X.select_dtypes(include='number').columns

X.loc[:, columns_number] = MinMaxScaler().fit_transform(X[columns_number])

X = pd.get_dummies(X, columns=x_columns[X.dtypes == 'object'])

X
# 目的変数yを設定

y = merged_teams['Res']

y
# 学習アルゴリズムとパrメータ候補を定義

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

    'instance': LogisticRegression(max_iter=200, n_jobs=-1),

    'params': [

            {'penalty': ['l2'], 'C': [0.1, 0.5, 1, 5, 10]},

            {'penalty': ['l1'], 'solver': ['liblinear', 'saga'], 'C': [0.1, 0.5, 1, 5, 10]},

            {'penalty': ['elasticnet'], 'C': [0.1, 0.5, 1, 5, 10], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]}

        ]

}
# 学習し、最適なパラメータを探索

for clf_name, clf in clfs.items():

    print('<{}>'.format(clf_name))

    print('  training ...')

    

    # 学習

    gs = GridSearchCV(clf['instance'], param_grid=clf['params'], cv=5, n_jobs=-1)

    gs.fit(X, y)

    

    print('  best_score: {:.3f}'.format(gs.best_score_))

    print('  best_params: {}'.format(gs.best_params_))

    

    # 最適なパラメータを記録

    clfs[clf_name]['best_params'] = gs.best_params_
# 最適なパラメータ値で、全データで学習してみる

for clf_name, clf in clfs.items():

    clf['best_estimator'] = clf['instance'].set_params(**clf['best_params']).fit(X, y)
[clf['best_estimator'] for clf in clfs.values()]
# 各モデルの予測結果から多数決を採用するソフト分類器を作成

vote = VotingClassifier(

    estimators=[(clf_name, clf['best_estimator']) for clf_name, clf in clfs.items()], 

    voting='soft',

    n_jobs=-1

)

vote.fit(X, y)



# 作成したソフト分類器を、clfsに追加

clfs['Vote'] = {}

clfs['Vote']['best_estimator'] = vote
# 各モデルとソフト分類器で、予測結果を比較する

# randomforestだけ精度が異常に高いのは、多分過学習してる。。。

for clf_name, clf in clfs.items():

    score = accuracy_score(y, clf['best_estimator'].predict(X))

    print(clf_name, score)
# 各モデルの予測結果をDataFrameに整形（pp: predict_proba, p: predict）

predict = pd.DataFrame(

    {

        **{'pp_' + clf_name: clf['best_estimator'].predict_proba(X)[:, 1] for clf_name, clf in clfs.items()},

        **{'p_' + clf_name: clf['best_estimator'].predict(X) for clf_name, clf in clfs.items()}

    },

    index=X.index)

predict
# 提出ファイルにカラムを追加し、予測結果の値を上書き

for column in predict.columns:

    sub[column] = 0.5

    

mask = [idx for idx in sub.index if idx in X.index]

sub.loc[mask, predict.columns] = predict.loc[mask, predict.columns]

sub
# 予測結果をファイル出力

for column in predict.columns:

    sub[['ID', column]].rename(columns={column: 'pred'}).to_csv('{}.csv'.format(column), index=False)