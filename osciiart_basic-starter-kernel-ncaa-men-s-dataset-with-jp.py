# 必要なライブラリとデータの読み込み
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# データ読み込み
data_dir = '../input/'
df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'NCAATourneyCompactResults.csv')
df_seeds.head() # 各年度のシード順とチームIDの対応表
"""
レギュラーシーズンの結果
Season: 年度
DayNum: 節　実際の日付は"Seasons.csv"参照
WTeamID: 勝利チーム
WScore: 勝利点数
LTeamID: 敗北チーム
LScore: 敗北スコア
WLoc: 試合会場が勝利チームにとってHomeかAwayかNeutralか
NumOT: 延長戦の回数
"""
df_tour.head()

# シード情報をintに変換
def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
df_seeds.head()
# 使う項目だけに削る
df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)
df_tour.head()
"""
WSeed: WTeamIDのチームのシード順
LSeed: LTeamIDのチームのシード順
SeedDiff: シード差 = WSeed - LSeed
"""
df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
# Dataframe: df_tourとdf_winseedsをColumns名: Season, WTeamIDが一致するように結合
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
df_concat.head()
df_winseeds.iloc[:100]
"""
SeedDiffとResult=1(勝ち)のセットと
-SeedDiffとResult=0(負け)のセットを
連結する
SeedDiffを反転する = 勝ったチームと負けたチームを入れ替えることなのでResultが反転する
"""
df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_concat['SeedDiff']
df_wins['Result'] = 1

df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['Result'] = 0

df_predictions = pd.concat((df_wins, df_losses))
df_predictions.head()
"""
SeedDiffを説明変数(X_train), Resultをアウトカム(y_train)として
"""
X_train = df_predictions.SeedDiff.values.reshape(-1,1)
y_train = df_predictions.Result.values
X_train, y_train = shuffle(X_train, y_train)
"""
ロジスティック回帰をする．
L2正則化パラメータのCをグリッドサーチする．
1/2Cが正則化項の係数となる
"""
logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
print('Best log_loss: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_['C']))
"""
XとYの関係の可視化
それっぽいグラフが得られる
"""
X = np.arange(-10, 10).reshape(-1, 1)
preds = clf.predict_proba(X)[:,1]

plt.plot(X, preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')
"""
予測する対戦組み合わせを読み込む
2014-2017の各年度の大会の出場校の全部の組み合わせの試合を予測する．
すなわち 4年 x 68チームから2チーム選ぶ組み合わせ = 4 x 68 x 67 / 2 = 9112
"""

df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """
    Return a tuple with ints `year`, `team1` and `team2`.
    IDが 年_チーム1_チーム2 の形式になってるのでそれを3つに切り分ける．
    """
    return (int(x) for x in ID.split('_'))
"""
予測する試合のdiff_seedを計算する
"""
X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed
# 学習モデルで予測する
preds = clf.predict_proba(X_test)[:,1]

clipped_preds = np.clip(preds, 0.05, 0.95) # loglossのペナルティ回避のため(0.05-0.95)の範囲に丸める
df_sample_sub.Pred = clipped_preds
df_sample_sub.head()
# 出力する
df_sample_sub.to_csv('logreg_seed_starter.csv', index=False)