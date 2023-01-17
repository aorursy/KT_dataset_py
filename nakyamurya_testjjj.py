import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
#import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option('display.max_columns', 100)
%matplotlib inline
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
print(train.shape)
print(test.shape)
train.head()
train_col = train.columns
test_col = test.columns
#trainとtestで一致しないカラムチェック
set(train_col) ^ set(test_col)
#ある特定のmatchでの順位を見てみる
match_id = 'a10357fd1a4a91'
game = train[train.matchId == match_id]
#順位で並べ替え
game.sort_values(['winPlacePerc'], ascending=False)
#matchTypeの数
train['matchType'].unique()
#まずはsoloだけで見てみる
df_solo = train[train.matchType == 'solo']
df_solo
df_solo.corr()
correlations = df_solo.corr()
mask = np.array(correlations)
mask[np.tril_indices_from(correlations)] = False
# Create color map ranging between two colors
cmap = sns.diverging_palette(220, 10, as_cmap=True)

fig, ax = plt.subplots(figsize=(20,20))
sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5,
            annot=True, cbar_kws={"shrink": .75},mask=mask)
plt.show();
#試しにwinPlacePercと相関の高い係数4つを選んで機械学習してみる
fig,(ax1,ax2,ax3,ax4) = plt.subplots(ncols=4)
fig.set_size_inches(12, 5)
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)
ax3.set_ylim(0,1)
ax4.set_ylim(0,1)


sns.regplot(x="boosts", y="winPlacePerc", data=df_solo,ax=ax1)
sns.regplot(x="kills", y="winPlacePerc", data=df_solo,ax=ax2)
sns.regplot(x="weaponsAcquired", y="winPlacePerc", data=df_solo,ax=ax3)
sns.regplot(x="walkDistance", y="winPlacePerc", data=df_solo,ax=ax4)

from sklearn import linear_model
clf = linear_model.LinearRegression()

# 説明変数
df_solo_x = df_solo[['boosts','kills','weaponsAcquired','walkDistance']]
X = df_solo_x.as_matrix()
 
# 目的変数
Y = df_solo['winPlacePerc'].as_matrix()
 
# 予測モデルを作成
clf.fit(X, Y)
 
# 偏回帰係数
print(pd.DataFrame({"Name":df_solo_x.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') )
 
# 切片 (誤差)
print(clf.intercept_)
#作ったモデルで予測
pred = clf.predict(X)
from sklearn.metrics import mean_absolute_error #MeanAbsoluteErrorで評価
mean_absolute_error(df_solo['winPlacePerc'], pred)
#Testデータで予測
pred= clf.predict(test[['boosts','kills','weaponsAcquired','walkDistance']])
Id = np.array(test['Id'])
my_solution = pd.DataFrame(pred, Id, columns = ["winPlacePerc"])
my_solution.to_csv('my_solution.csv', index_label = ['Id'])
