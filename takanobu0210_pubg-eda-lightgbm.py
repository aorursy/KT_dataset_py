import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
# データの読み込み
df_train = reduce_mem_usage(pd.read_csv('../input/train_V2.csv'))
df_test = reduce_mem_usage(pd.read_csv('../input/test_V2.csv'))
# データの形状確認
print('train : {}'.format(df_train.shape))
print('test : {}'.format(df_test.shape))
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
percent_data = percent.head()
percent_data.plot(kind='bar', figsize=(8,6), fontsize=10)
plt.xlabel('Columns', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Total Missing Value (%) in Train', fontsize=20)
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head()
percent_data = percent.head()
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Value (%) in Test", fontsize = 20)
df_train.dropna(axis=0, inplace=True)
# 削除されているか確認
df_train.shape
# df_train.isnull().sum()
# ヒートマップを表示してみる
k = 10 # 表示する特徴量の数
corrmat = df_train.corr()
cols = corrmat.nlargest(k, 'winPlacePerc').index # リストの最大値から順にk個の要素の添字(index)を取得
# df_train[cols].head()
cm = np.corrcoef(df_train[cols].values.T) # 相関関数行列を求める ※転置が必要
# cm[:5]
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(16, 12))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
# 徒歩での移動距離と目的変数
df_train.plot(x='winPlacePerc', y='walkDistance', kind='scatter', figsize=(8,6))
# boostアイテムと目的変数
f, ax = plt.subplots(figsize=(14,6))
fig = sns.boxplot(x='boosts', y='winPlacePerc', data=df_train)
fig.axis(ymin=0, ymax=1)
df_train[df_train['boosts'] >= 24].head()
# 取得した武器の数と目的変数
df_train.plot(x='winPlacePerc',y='weaponsAcquired', kind='scatter', figsize = (8,6))
# 与えたダメージ量と目的変数
df_train.plot(x='winPlacePerc', y='damageDealt', kind='scatter', figsize=(8,6))
# 回復アイテムと目的変数
df_train.plot(x='winPlacePerc',y='heals', kind='scatter', figsize = (8,6))
# 最長射殺距離と目的変数
df_train.plot(x='winPlacePerc',y='longestKill', kind='scatter', figsize = (8,6))
# 倒した数と目的変数
df_train.plot(x='winPlacePerc',y='kills', kind='scatter', figsize = (8,6))
headshot = df_train[['kills', 'winPlacePerc', 'headshotKills']]
headshot['headshotrate'] = headshot['headshotKills'] / headshot['kills']
headshot.corr() # 相関を確認
df_train['headshotrate'] = df_train['headshotKills']/df_train['kills']
df_test['headshotrate'] = df_test['headshotKills']/df_test['kills']
del headshot # メモリ解放
killStreak = df_train[['kills','winPlacePerc','killStreaks']]
killStreak['killStreakrate'] = killStreak['killStreaks']/killStreak['kills']
killStreak.corr() # 相関を確認
df_train['killStreakrate'] = -(df_train['killStreaks'] / df_train['kills'])
df_test['killStreakrate'] = -(df_test['killStreaks'] / df_test['kills'])
del killStreak
# ハッカーポイントの特徴量を作成
df_train['hacker_pt'] = 0
df_test['hacker_pt'] = 0
# pandasの最大表示列数を広げておく（ここでは50列を指定）
pd.set_option('display.max_columns', 50)
df_train['total_Distance'] = df_train['rideDistance'] +df_train['walkDistance'] + df_train['swimDistance']
df_test['total_Distance'] = df_test['rideDistance'] + df_test['walkDistance'] + df_test['swimDistance']
df_train[(df_train['winPlacePerc'] == 1) & (df_train['total_Distance'] < 100)].head() # 表示
df_train['headshotrate'] = df_train['headshotrate'].fillna(0)
df_train['killStreakrate'] = df_train['killStreakrate'].fillna(0)
df_test['headshotrate'] = df_test['headshotrate'].fillna(0)
df_test['killStreakrate'] = df_test['killStreakrate'].fillna(0)
# df_train.isnull().sum()
df_train[(df_train['winPlacePerc'] == 1) & (df_train['total_Distance'] < 100)].describe()
df_train['hacker_pt'][(df_train['heals'] + df_train['boosts'] < 1) & (df_train['total_Distance'] < 100) & (df_train['kills'] > 20)] = 1
df_test['hacker_pt'][(df_test['heals'] + df_test['boosts'] < 1) & (df_test['total_Distance'] < 100) & (df_test['kills'] > 20)] = 1
df_train[(df_train['kills'] > 10) &(df_train['weaponsAcquired'] >= 10) & (df_train['total_Distance'] == 0)].describe()
df_train['hacker_pt'][(df_train['kills'] > 10) &(df_train['weaponsAcquired'] >= 10) & (df_train['total_Distance'] == 0)] += 1
df_test['hacker_pt'][(df_test['kills'] > 10) &(df_test['weaponsAcquired'] >= 10) & (df_test['total_Distance'] == 0)] += 1
df_train[df_train['longestKill'] >= 1000].describe()
df_train['hacker_pt'][df_train['longestKill'] >= 1000] += 1
df_test['hacker_pt'][df_train['longestKill'] >= 1000] += 1
df_train.sort_values('hacker_pt', ascending=False).head()
kills = df_train[['assists','winPlacePerc','kills']]
kills['kills_assists'] = (kills['kills'] + kills['assists'])
kills.corr()
df_train['kills_assists'] = df_train['kills'] + df_train['assists']
df_test['kills_assists'] = df_test['kills'] + df_test['assists']
del df_train['kills']
del df_test['kills']
del df_train['assists']
del df_test['assists']
del kills
# メモリの開放を行う
import gc
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
gc.collect()
# メモリ食ってる変数を確認してみる
import sys

print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
print(" ------------------------------------ ")
for var_name in dir():
    if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 1000: #ここだけアレンジ
        print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))
# 不要な変数は削除する
del missing_data
del percent
del total
gc.collect()
# マッチID, グループIDごとのサイズ、平均値、最大値、最小値を取得
df_train_size = df_train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
df_test_size = df_test.groupby(['matchId','groupId']).size().reset_index(name='group_size')
df_train_mean = df_train.groupby(['matchId','groupId']).mean().reset_index()
df_test_mean = df_test.groupby(['matchId','groupId']).mean().reset_index()
# df_train_max = df_train.groupby(['matchId','groupId']).max().reset_index()
# df_test_max = df_test.groupby(['matchId','groupId']).max().reset_index()
# df_train_min = df_train.groupby(['matchId','groupId']).min().reset_index()
# df_test_min = df_test.groupby(['matchId','groupId']).min().reset_index()
df_train_match_mean = df_train.groupby(['matchId']).mean().reset_index()
df_test_match_mean = df_test.groupby(['matchId']).mean().reset_index()
df_train = pd.merge(df_train, df_train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del df_train_mean
del df_test_mean
df_train = pd.merge(df_train, df_train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
df_test = pd.merge(df_test, df_test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del df_train_match_mean
del df_test_match_mean
df_train = pd.merge(df_train, df_train_size, how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_size, how='left', on=['matchId', 'groupId'])
del df_train_size
del df_test_size
gc.collect()
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
gc.collect()
import os
import time
import warnings
warnings.filterwarnings("ignore")
# data manipulation
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd

# plot
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
train_columns = list(df_test.columns)

train_idx = df_train.Id
test_idx = df_test.Id

train_columns.remove("Id")
train_columns.remove("matchId")
train_columns.remove("groupId")
x_train = df_train[train_columns]
x_test = df_test[train_columns]
y_train = df_train["winPlacePerc"].astype('float')
x_train.head()
x_test.head()
y_train.head()
encoded_train = pd.get_dummies(x_train.matchType, prefix=x_train.matchType.name ,prefix_sep="_")
encoded_test = pd.get_dummies(x_test.matchType, prefix=x_test.matchType.name ,prefix_sep="_")
encoded_train.head()
x_train = x_train.merge(encoded_train, right_index=True, left_index=True)
x_test = x_test.merge(encoded_test, right_index=True, left_index=True)
x_train.head()
del x_train['matchType']
del x_test['matchType']
del df_train; del df_test
gc.collect()
# LightGBM
folds = KFold(n_splits=3,random_state=6)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

start = time.time()
valid_score = 0
importances = pd.DataFrame()

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    trn_x, trn_y = x_train.iloc[trn_idx], y_train[trn_idx]
    val_x, val_y = x_train.iloc[val_idx], y_train[val_idx]    
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)
    
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':10000, 'early_stopping_rounds':100,
              "num_leaves" : 30, "learning_rate" : 0.3, "bagging_fraction" : 0.9,
               "bagging_seed" : 0}
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 

    #imp_df = pd.DataFrame()
    #imp_df['feature'] = train_columns
    #imp_df['gain'] = lgb_model.booster_.feature_importance(importance_type='gain')
    
    #imp_df['fold'] = fold_ + 1
    #importances = pd.concat([importances, imp_df], axis=0, sort=False)    
    
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    oof_preds[oof_preds>1] = 1
    oof_preds[oof_preds<0] = 0
    sub_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / folds.n_splits
    sub_pred[sub_pred>1] = 1 # should be greater or equal to 1
    sub_pred[sub_pred<0] = 0 
    sub_preds += sub_pred
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, mean_absolute_error(val_y, oof_preds[val_idx])))
    valid_score += mean_absolute_error(val_y, oof_preds[val_idx])
print('Done')
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["winPlacePerc"] = sub_preds
test_pred.columns = ["Id", "winPlacePerc"]
test_pred.to_csv("lgb_base_model.csv", index=False) # submission
print('Done')
