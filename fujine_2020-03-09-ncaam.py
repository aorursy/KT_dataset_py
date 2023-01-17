import os



import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_validate

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
# 提出用ファイルを取得

sub = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')

id_splited = sub['ID'].str.split('_', expand=True).astype(int).rename(columns={0: 'Season', 1: 'Team1', 2: 'Team2'})

sub = pd.concat([sub, id_splited], axis=1).set_index(['Season', 'Team1', 'Team2']).sort_index()

sub
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

conferences = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamConferences.csv')

conferences = pd.concat([conferences.query('Season == @season and TeamID in @teams') for season, teams in tourney_teams.items()])

conferences = conferences.set_index(['Season', 'TeamID']).sort_index()

conferences
# コーチ名を取得

# シーズン期間中にコーチを変更したチームがあるため、今回は「トーナメント開始時点でコーチだった

# 人物が、トーナメント結果に強い影響を与える」と仮定し、LastDayNumが154のコーチを使用する。

coaches = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeamCoaches.csv')

coaches = pd.concat([coaches.query('Season == @season and TeamID in @team') for season, team in tourney_teams.items()])

#coaches[(coaches['FirstDayNum'] != 0) | (coaches['LastDayNum'] != 154)] 

coaches = coaches[coaches['LastDayNum'] == 154].set_index(['Season', 'TeamID']).sort_index().loc[:, ['CoachName']]

coaches
# NCAAの初回出場年を取得し、初回出場年から現在までの年数を計算

teams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv', usecols=['TeamID', 'FirstD1Season'])

teams['First'] = 2020 - teams.pop('FirstD1Season')

teams = pd.concat([teams.query('TeamID in @team').assign(Season=season) for season, team in tourney_teams.items()])

teams = teams.set_index(['Season', 'TeamID']).sort_index()

teams
# 各シーズンのシードを取得

seeds = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')

seeds = pd.concat([seeds.query('Season == @season and TeamID in @teams') for season, teams in tourney_teams.items()])

seeds = seeds.set_index(['Season', 'TeamID']).sort_index()

seeds['Region'] = seeds['Seed'].str[0]

seeds['Number'] = seeds['Seed'].str[1:3].astype(int)

del seeds['Seed']

seeds
# 上記取得データを結合

ctcs = pd.concat([coaches, teams, conferences, seeds], axis=1)

ctcs
# NCAAMトーナメントの勝敗結果を取得

result = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')

result = result[result['Season'] >= 2015].set_index(['Season', 'WTeamID', 'LTeamID'])

result
# 各種データと勝敗結果を結合

merged_teams = pd.concat([ctcs.loc[[(season, wteam), (season, lteam)], :] for season, wteam, lteam, in result.index])



team1 = merged_teams.iloc[::2, :].reset_index('TeamID') # teams winned

team2 = merged_teams.iloc[1::2, :].reset_index('TeamID') # teams losed



merged_teams = pd.concat([

    pd.concat([team1.add_suffix('1'), team2.add_suffix('2')], axis=1).assign(Res=1),

    pd.concat([team2.add_suffix('1'), team1.add_suffix('2')], axis=1).assign(Res=0),

]).reset_index().set_index(['Season', 'TeamID1', 'TeamID2'])

merged_teams
# データを、説明変数Xと目的変数yに分離

columns = merged_teams.columns

X = merged_teams[columns[columns != 'Res']]

y = merged_teams['Res']
X
y
# Xを数値化、正規化

for c in X.select_dtypes(include='number').columns:

    X[c] = MinMaxScaler().fit_transform(X[c].to_numpy().reshape(-1,1))

X = pd.get_dummies(X, columns=X.columns[X.dtypes == 'object'])

X
# RandomForestで学習＆予測精度を出力

clf = RandomForestClassifier(n_estimators=200)

score = cross_validate(clf, X, y, cv=3, return_estimator=True)

score['test_score']
# 各変数の重要度を可視化

importances = pd.Series(score['estimator'][1].feature_importances_, index=X.columns)

_ = importances.sort_values(ascending=False).head(30).plot(kind='bar', grid=True, figsize=(20, 5))
# 次にSVCで学習＆予測精度を出力

clf = SVC(kernel='rbf', probability=True)

score = cross_validate(clf, X, y, cv=3, return_estimator=True)

score['test_score']
# SVCのkernelを'linear'にすれば、以下コードで各説明変数の寄与率を出力できる

#importances = pd.Series(score['estimator'][2].coef_.flatten(), index=X.columns)

#_ = importances.abs().sort_values(ascending=False).head(30).plot(kind='bar', grid=True, figsize=(20, 5))
# SVCで作成した予測モデルから予測結果を出力し、DataFrame型に加工

best_estimator = score['estimator'][score['test_score'].argmax()]

pred = pd.DataFrame(best_estimator.predict_proba(X), index=X.index, columns=best_estimator.classes_)

pred
# 予測結果を提出用データに上書き

sub['Pred'] = 0.5

for idx in sub.index:

    if idx in X.index:

        sub.loc[idx, 'Pred'] = pred.loc[idx, 1]

        

_ = sub['Pred'].plot(kind='hist', grid=True, bins=50, logy=True)

_ = sub[sub['Pred'] != 0.5]['Pred'].plot(kind='hist', bins=50, grid=True)
# 実際のトーナメント組合せだけで学習すると、全組合せの6%しか無いんだな。

# まあ、トーナメントに無い組合せはどんな数値を入れても実際のスコアには影響しないんだろうけど

len(X) / len(sub) * 100
# 提出用データをファイル出力

sub.to_csv('submissoin.csv', index=False)

pd.read_csv('submissoin.csv')