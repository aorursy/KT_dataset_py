# ライブラリを読み込む
import pandas as pd
import numpy as np
from sklearn import *
import glob
# データを読み込む
datafiles = sorted(glob.glob('../input/**.csv')) # 全データのパスを取得
# 全データをpd.Dataframeとして読み込んでdictionaryにまとめる
datafiles = {file.split('/')[-1].split('.')[0]: pd.read_csv(file, encoding='latin-1') for file in datafiles}
# 全データをリストアップ
for i, key in enumerate(datafiles):
    print(i, key)
"""
SecondaryTourneyというカラムを追加して 'NCAA"もしくは'Regular'の値を与える
SecondaryTourneyはNCAAトーナメント以外のトーナメントのことでNIT, CBI, CIT, V16の４つがある．
ここにNCAA, Regularを追加して試合がどの大会に属するものかを表すカラムとして扱う．
"""
datafiles['NCAATourneyCompactResults']['SecondaryTourney'] = 'NCAA'
datafiles['NCAATourneyDetailedResults']['SecondaryTourney'] = 'NCAA'
datafiles['RegularSeasonCompactResults']['SecondaryTourney'] = 'Regular'
datafiles['RegularSeasonDetailedResults']['SecondaryTourney'] = 'Regular'
### Presets
# カテゴリ型のカラムを数値に置き換える準備
WLoc = {'A': 1, 'H': 2, 'N': 3}
SecondaryTourney = {'NIT': 1, 'CBI': 2, 'CIT': 3, 'V16': 4, 'Regular': 5 ,'NCAA': 6}
# NCAAトーナメントとレギュラーリーグの試合情報を連結

# コンパクト版
# NCAA, レギュラーシーズンの試合情報を連結
games = pd.concat((datafiles['NCAATourneyCompactResults'],datafiles['RegularSeasonCompactResults']), axis=0, ignore_index=True)
# さらにSecondaryTourney () の試合情報を連結
games = pd.concat((games,datafiles['SecondaryTourneyCompactResults']), axis=0, ignore_index=True)

# 詳細版
#games = pd.concat((datafiles['NCAATourneyDetailedResults'],datafiles['RegularSeasonDetailedResults']), axis=0, ignore_index=True)

games.reset_index(drop=True, inplace=True) # indexをリセット
# 試合会場情報をH(ome), A(way), N(eutral)から1, 2, 3 に変換
games['WLoc'] = games['WLoc'].map(WLoc)
# 大会形式情報をNIT, CBI, CIT, V16, Regular, NCAAから1, 2, 3, 4, 5, 6 に変換
games['SecondaryTourney'] = games['SecondaryTourney'].map(SecondaryTourney)
print("games.shape", games.shape)
games.head()
###Add Ids
# カラム ID, IDTeam, Team1, Team2, IDTeam1, IDTeam2 を追加
# Team1, Team2は勝利チームと敗北チームをソートしたものなので入れ替わっている場合有り
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
print("games.shape", games.shape)
games.head()
###Add Seeds
# シード情報を読み込みIDTeamsとひも付けた形に変換
seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in datafiles['NCAATourneySeeds'].values}
for key in sorted(seeds.keys())[:10]:
    print(key, seeds[key])
### Add 2018
# 2018年のシード順位を追加する．2018年のシード順位は2017年と同じとする．
if 2018 not in datafiles['NCAATourneySeeds']['Season'].unique():
    seeds = {**seeds, **{k.replace('2017_','2018_'):seeds[k] for k in seeds if '2017_' in k}}
print("2017年のDuke大学のシード順位:", seeds['2017_1181'], "2018年のDuke大学のシード順位:", seeds['2018_1181'])
# シード順位情報のカラムを追加. 情報がないチームは0を代入
games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)
games.head()
### Additional Features & Clean Up
# 特徴量を追加する
games['ScoreDiff'] = games['WScore'] - games['LScore'] # 点差
# 試合結果 (チーム1 = 勝利チーム なら 1 (勝ち))
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)
# 点差の絶対値 (負け試合だと点差がマイナスになっているので)
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed'] # シード順位差
games = games.fillna(-1) # NAを-1で埋める
games.head()
### Test Set
#　テストデータ作成
sub = datafiles['SampleSubmissionStage1']
sub['WLoc'] = 3 #N NCAAの試合会場は必ず中立
sub['SecondaryTourney'] = 6 #NCAA
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])
sub['Season'] = sub['Season'].astype(int)
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed'] 
sub.head()
sub.tail()
### 2017年分を予測
season = 2017
print(season)
# 前年度以前のNCAAのデータを抜き出す
x1 = games[((games['Season']<int(season)) & (games['SecondaryTourney']==6))]
# 同年度以前のNCAA以外のデータを抜き出し追加
x1 = pd.concat((x1,games[((games['Season']<int(int(season)+1)) & (games['SecondaryTourney']!=6))]), axis=0, 
               ignore_index=True)
print(x1.shape)
x1.tail()
# 同年度のNCAAのデータを抜き出す
x2 = games[((games['Season']==int(season)) & (games['SecondaryTourney']==6))]
print(x2.shape)
x2.head()
test = sub[sub['Season']==season]
print(test.shape)
test.head()
sdn = x1.groupby(['IDTeams'], as_index=False)[['ScoreDiffNorm']].mean() # 同じ対戦カードの過去の点差の平均
sdn[:10]
test = pd.merge(test, sdn, how='left', on=['IDTeams'])
test['ScoreDiffNorm'] = test['ScoreDiffNorm'].fillna(0.)
test.head()
### Interactions
# Interという特徴量を作る．ここの意味がよく分かっていない
# 直近2年間の試合でチーム1，2両方が戦ったことのある相手に対してチーム1の勝ち数 - 負け数？
"""
チーム1, チーム2, 年度, 結果だけを抜き出す
チーム2を先にした版をまず用意
チーム2が先に来るので結果を反転
"""
inter = games[['IDTeam2','IDTeam1','Season','Pred']].rename(columns={'IDTeam2':'Target','IDTeam1':'Common'})
inter['Pred'] = inter['Pred'] * -1
print(inter.shape)
inter.head()
# チーム1を先にした版を追加
inter = pd.concat((inter,games[['IDTeam1','IDTeam2','Season','Pred']].rename(columns={'IDTeam1':'Target','IDTeam2':'Common'})), axis=0, ignore_index=True).reset_index(drop=True)
print(inter.shape)
inter.head()
# 3年以上前のデータは捨てる
#Only two years back and current regular season
inter = inter[((inter['Season']<=int(season)) & (inter['Season']>int(season)-2))]
print(inter.shape)
inter.head()
inter = pd.merge(inter, inter, how='inner', on=['Common','Season'])  # ここがわからん
print(inter.shape)
inter.head()
inter = inter[inter['Target_x'] != inter['Target_y']]
print(inter.shape)
inter.head()
# inter['ID'] = inter.apply(lambda r:
#                           '_'.join(map(str, [r['Season']+1, 
#                                              r['Target_x'].split('_')[1],
#                                              r['Target_y'].split('_')[1]])), axis=1)
inter['IDTeams'] = inter.apply(lambda r: 
                               '_'.join(map(str, [r['Target_x'].split('_')[1],
                                                  r['Target_y'].split('_')[1]])), axis=1)
inter = inter[['IDTeams','Pred_x']]
print(inter.shape)
inter.head()
inter = inter.groupby(['IDTeams'], as_index=False)[['Pred_x']].sum()
inter = {k:int(v) for k, v in inter.values}
for key in sorted(inter.keys())[:10]:
    print(key, inter[key])
# x1 ,x2, test にInterカラムを追加
x1['Inter'] = x1['IDTeams'].map(inter).fillna(0)
x2['Inter'] = x2['IDTeams'].map(inter).fillna(0)
test['Inter'] = test['IDTeams'].map(inter).fillna(0)
print(x1.shape)
x1.head()
# 説明変数を選択
col = [c for c in x1.columns if c not in [
    'ID', 
    'Team1',
    'Team2', 
    'IDTeams',
    'IDTeam1',
    'IDTeam2',
    'Pred',
    'DayNum', 
    'WTeamID', 
    'WScore', 
    'LTeamID', 
    'LScore', 
    'NumOT', 
    'ScoreDiff']]
col
sorted(x1['Inter'].unique())
"""
HuberRegressorモデルをトレーニング
HuberRegressorはoutlierにつよいregressorとのこと
"""

reg = linear_model.HuberRegressor()
reg.fit(x1[col], x1['Pred'])
pred = reg.predict(x2[col]).clip(0.05, 0.95)
print('Log Loss:', metrics.log_loss(x2['Pred'], pred))
test['Pred'] = reg.predict(test[col])
### Add Validation
results = []
for season in sub['Season'].unique(): # 年度ごとに処理
    print(season)
    # 前年度以前のNCAAのデータを抜き出す
    x1 = games[((games['Season']<int(season)) & (games['SecondaryTourney']==6))]
    # 同年度以前のNCAA以外のデータを抜き出し追加
    x1 = pd.concat((x1,games[((games['Season']<int(int(season)+1)) & (games['SecondaryTourney']!=6))]), axis=0, 
                   ignore_index=True)
    # 同年度のNCAAのデータを抜き出す
    x2 = games[((games['Season']==int(season)) & (games['SecondaryTourney']==6))]
    test = sub[sub['Season']==season]

    sdn = x1.groupby(['IDTeams'], as_index=False)[['ScoreDiffNorm']].mean() # 同じ対戦カードの過去の点差の平均
    test = pd.merge(test, sdn, how='left', on=['IDTeams'])
    test['ScoreDiffNorm'] = test['ScoreDiffNorm'].fillna(0.)
    
    #Interactions
    inter = games[['IDTeam2','IDTeam1','Season','Pred']].rename(columns={'IDTeam2':'Target','IDTeam1':'Common'})
    inter['Pred'] = inter['Pred'] * -1
    inter = pd.concat((inter,games[['IDTeam1','IDTeam2','Season','Pred']].rename(columns={'IDTeam1':'Target','IDTeam2':'Common'})), axis=0, ignore_index=True).reset_index(drop=True)
    inter = inter[((inter['Season']<=int(season)) & (inter['Season']>int(season)-2))] #Only two years back and current regular season
    inter = pd.merge(inter, inter, how='inner', on=['Common','Season'])
    inter = inter[inter['Target_x'] != inter['Target_y']]
    #inter['ID'] = inter.apply(lambda r: '_'.join(map(str, [r['Season']+1, r['Target_x'].split('_')[1],r['Target_y'].split('_')[1]])), axis=1)
    inter['IDTeams'] = inter.apply(lambda r: '_'.join(map(str, [r['Target_x'].split('_')[1],r['Target_y'].split('_')[1]])), axis=1)
    inter = inter[['IDTeams','Pred_x']]
    inter = inter.groupby(['IDTeams'], as_index=False)[['Pred_x']].sum()
    inter = {k:int(v) for k, v in inter.values}
    
    x1['Inter'] = x1['IDTeams'].map(inter).fillna(0)
    x2['Inter'] = x2['IDTeams'].map(inter).fillna(0)
    test['Inter'] = test['IDTeams'].map(inter).fillna(0)
    col = [c for c in x1.columns if c not in ['ID', 'Team1','Team2', 'IDTeams','IDTeam1','IDTeam2','Pred','DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'ScoreDiff']]
    
    reg = linear_model.HuberRegressor()
    reg.fit(x1[col], x1['Pred'])
    pred = reg.predict(x2[col]).clip(0.05, 0.95)
    print('Log Loss:', metrics.log_loss(x2['Pred'], pred))
    test['Pred'] = reg.predict(test[col])

    results.append(test)
results = pd.concat(results, axis=0, ignore_index=True).reset_index(drop=True)
#Testing for Sequence of Scoring
results = {k:float(v) for k,v in results[['ID','Pred']].values}
sub['Pred'] = sub['ID'].map(results).clip(0.05, 0.95).fillna(0.49)
sub[['ID','Pred']].to_csv('rh3p_submission.csv', index=False)