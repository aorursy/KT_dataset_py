%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import seaborn as sns
from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数
from pylab import rcParams
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats
df_kickstar = pd.read_csv("../1_data/ks-projects-201801.csv")[['ID','name','category','main_category','currency','deadline','goal','launched','pledged','state','backers','country','usd pledged','usd_pledged_real','usd_goal_real']]
# df_kickstar = pd.read_csv("../input/ks-projects-201801.csv")[['ID','name','category','main_category','currency','deadline','goal','launched','pledged','state','backers','country','usd pledged','usd_pledged_real','usd_goal_real']]

# 先程と似た中古住宅のデータ
display(df_kickstar.head())
df_kickstar.describe()
# 不要列削除
# df_kickstar = df_kickstar.drop(['ID','name','category','country','usd_pledged_real','usd_goal_real','main_category'], axis=1)
# df_kickstar = df_kickstar.drop(['ID','name','category','country','usd_pledged_real','usd_goal_real'], axis=1)
# df_kickstar = df_kickstar.drop(['ID','name','category','country','usd_pledged_real','usd_goal_real','backers','pledged'], axis=1)

# 列整理
# df_kickstar = df_kickstar.loc[:,['state','goal','usd pledged','deadline','launched','currency','main_category','usd_goal_real','category','country']]

# df_kickstar.head()
# 成功とそれ以外に分ける
display(df_kickstar['state'].value_counts())
# df_kickstar['state'] = df_kickstar['state'] == 'successful'
df_kickstar['state'] = df_kickstar['state'] == 'successful'
# deadlineとlaunchesから日数の列を作成
df_kickstar.dtypes
df_kickstar['deadline'] = pd.to_datetime(df_kickstar['deadline'], errors = 'coerce')
df_kickstar['launched'] = pd.to_datetime(df_kickstar['launched'], errors = 'coerce')

# 日数の列daysを追加
df_kickstar['days'] = (df_kickstar['deadline'] - df_kickstar['launched']).dt.days

df_kickstar = df_kickstar.drop(['deadline','launched'], axis=1)

df_kickstar.head()
display(df_kickstar['category'].value_counts())
display(df_kickstar['main_category'].value_counts())
pd.plotting.scatter_matrix(df_kickstar[['goal','usd_goal_real','days']], figsize=(10,10))
plt.show()
sns.heatmap(df_kickstar.corr())
plt.show()
df_kickstar.isnull().sum()
#欠損値削除
df_kickstar = df_kickstar.dropna()
df_kickstar.isnull().sum()
# df_kickstar_sum = df_kickstar.query('state == True')
# df_kickstar_sum = df_kickstar.groupby(['currency','state']).count().sort_values('goal',ascending=False)
#成功と失敗表示
df_mcate_all = df_kickstar[['main_category','state']].groupby(['main_category'],).count().sort_values('main_category',ascending=False).rename(columns={'state': 'allcnt'})
df_mcate_all['okcnt'] =  df_kickstar[['main_category','state']].query('state == True').groupby(['main_category']).count().sort_values('main_category',ascending=False)
df_mcate_all['succcnt'] = (df_mcate_all['okcnt'] / df_mcate_all['allcnt'] * 100).astype(str) + '%' 
display(df_mcate_all.sort_values('succcnt',ascending=False))
# currencyをダミーカラムに変更
df_dummy = pd.get_dummies(df_kickstar['currency'])

df_kickstar = pd.concat([df_kickstar.drop(['currency'],axis=1),df_dummy],axis=1)

# main_category
df_dummy = pd.get_dummies(df_kickstar['main_category'])

df_kickstar = pd.concat([df_kickstar.drop(['main_category'],axis=1),df_dummy],axis=1)

df_kickstar.head()
df_kickstar['others_currency'] = df_kickstar[['HKD','CHF','SGD','NOK','JPY']].sum(axis=1)
df_kickstar['others_main_category'] = df_kickstar[['Fashion','Photography','Dance','Crafts','Journalism']].sum(axis=1)
print(df_kickstar['others_currency'].head())
print(df_kickstar['others_main_category'].head())
df_kickstar = df_kickstar.drop(['HKD','CHF','SGD','NOK','JPY','Fashion','Photography','Dance','Crafts','Journalism'],axis=1)
# goal,daysの外れ値検出
goal = df_kickstar['goal']
days = df_kickstar['days']
usd_goal_real = df_kickstar['usd_goal_real']
plt.title("散布図")
plt.scatter(goal,days)
plt.scatter(usd_goal_real,days)

# goalの四分位を計算
#第一四分位数（=25パーセンタイル）
goal_q1 = stats.scoreatpercentile(goal, 25)
#第三四分位数（=75パーセンタイル）
goal_q3 = stats.scoreatpercentile(goal, 75)
#四分位範囲
goal_iqr = goal_q3 - goal_q1
# df_kickstar[['goal','days']].quantile(q=[0, 0.25, 0.5, 0.75, 1])

#外れ値の範囲を計算する
#第一四分位数 から四分位範囲（iqr*1.5）を引き算。
goal_iqr_min = goal_q1 - (goal_iqr) * 1.5
#第三四分位数 から四分位範囲（iqr*1.5）を足し算。
goal_iqr_max = goal_q3 + (goal_iqr) * 1.5

# daysの四分位を計算
#第一四分位数（=25パーセンタイル）
days_q1 = stats.scoreatpercentile(days, 25)
#第三四分位数（=75パーセンタイル）
days_q3 = stats.scoreatpercentile(days, 75)
#四分位範囲
days_iqr = days_q3 - days_q1
# df_kickstar[['goal','days']].quantile(q=[0, 0.25, 0.5, 0.75, 1])

#外れ値の範囲を計算する
#第一四分位数 から四分位範囲（iqr*1.5）を引き算。
days_iqr_min = days_q1 - (days_iqr) * 1.5
#第三四分位数 から四分位範囲（iqr*1.5）を足し算。
days_iqr_max = days_q3 + (days_iqr) * 1.5

# usd_goal_realの外れ値検出
# usd_goal_realの四分位を計算
#第一四分位数（=25パーセンタイル）
usd_goal_real_q1 = stats.scoreatpercentile(usd_goal_real, 25)
#第三四分位数（=75パーセンタイル）
usd_goal_real_q3 = stats.scoreatpercentile(usd_goal_real, 75)
#四分位範囲
usd_goal_real_iqr = usd_goal_real_q3 - usd_goal_real_q1
# df_kickstar[['usd_goal_real']].quantile(q=[0, 0.25, 0.5, 0.75, 1])

#外れ値の範囲を計算する
#第一四分位数 から四分位範囲（iqr*1.5）を引き算。
usd_goal_real_iqr_min = usd_goal_real_q1 - (usd_goal_real_iqr) * 1.5
#第三四分位数 から四分位範囲（iqr*1.5）を足し算。
usd_goal_real_iqr_max = usd_goal_real_q3 + (usd_goal_real) * 1.5

print(usd_goal_real,days_iqr_max,usd_goal_real_iqr_max)
# print(goal_iqr_max,days_iqr_max)
df_kickstar = df_kickstar[(df_kickstar['goal'] < goal_iqr_max) & (df_kickstar['days'] < days_iqr_max) & (df_kickstar['usd_goal_real'] < usd_goal_real_iqr_max)]
goal = df_kickstar['goal']
days = df_kickstar['days']
usd_goal_real = df_kickstar['usd_goal_real']

plt.scatter(goal,days)
plt.scatter(usd_goal_real,days)
df_kickstar[['goal','days','usd_goal_real']].hist(figsize=(10,10))
# goal、pledged 標準化
# stdsc = StandardScaler()
# df_kickstar["goal_std"] = stdsc.fit_transform(df_kickstar[["goal"]].values)
# df_kickstar["usd_goal_real_std"] = stdsc.fit_transform(df_kickstar[["usd_goal_real"]].values)
# df_kickstar["days_std"] = stdsc.fit_transform(df_kickstar[["days"]].values)
# df_kickstar["usd_pledged_std"] = stdsc.fit_transform(df_kickstar[["usd pledged"]].values)

# goal、pledged 正規化
# normsc = MinMaxScaler()
# df_kickstar["goal_std"] = normsc.fit_transform(df_kickstar[["goal"]].values)
# df_kickstar["usd_goal_real_std"] = normsc.fit_transform(df_kickstar[["usd_goal_real"]].values)
# df_kickstar["days_std"] = normsc.fit_transform(df_kickstar[["days"]].values)
# df_kickstar["usd_pledged_std"] = normsc.fit_transform(df_kickstar[["usd pledged"]].values)

# df_kickstar[['goal_std','usd_goal_real_std','days_std']].hist(figsize=(10,10))
# usd_goal_realが0付近にデータが多いのでBox-cox変換をして正規分布に近づける
# 0があるとエラーになるので、全体に0.00000001を足す
# df_kickstar['usd_goal_real_std'], lmbda = stats.boxcox(df_kickstar['usd_goal_real_std'] + 0.00000001 )
# df_kickstar['usd_goal_real_std'].hist(figsize=(10,10))
# pd.plotting.scatter_matrix(df_kickstar[['goal_std','usd_pledged_std']], figsize=(10,10))
# plt.show()
# 不要カラム削除
# df_kickstar = df_kickstar.drop(['ID','name','category','country','usd_pledged_real','backers','pledged','usd pledged','usd_goal_real','goal','days'], axis=1)
df_kickstar = df_kickstar.drop(['ID','name','category','country','usd_pledged_real','backers','pledged','usd pledged'], axis=1)
df_kickstar.columns
# 相関係数をヒートマップにして可視化
sns.heatmap(df_kickstar.corr())
plt.show()
# 今回はgoalを削除
df_kickstar = df_kickstar.drop(['goal'], axis=1)
# usd_goal_real削除用
# df_kickstar = df_kickstar.drop(['usd_goal_real_std'], axis=1)
y = df_kickstar['state'].values
X = df_kickstar.drop('state', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#標準化と正規化
# テストデータ、訓練データ標準化
stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

# テストデータ、訓練データ正規化
normsc = MinMaxScaler()
normsc.fit(X_train)
X_train_norm = normsc.transform(X_train)
X_test_norm = normsc.fit_transform(X_test)


clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234)
# 標準化で学習
# clf.fit(X_train_std, y_train)

# 正規化で学習
clf.fit(X_train_norm, y_train)

# ラベルを予測
y_est = clf.predict(X_test_std)

# 対数尤度を表示
print('対数尤度 = {:.3f}'.format(- log_loss(y_test, y_est)))

# 正答率を表示
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test, y_est)))
# 予測値と正解のクロス集計
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_est), 
                        index=['正解 = クラウドファンディング成功', '正解 = クラウドファンディング失敗'], 
                        columns=['予測 = クラウドファンディング成功', '予測 = クラウドファンディング失敗'])
conf_mat
# 正答率を計算
accuracy = accuracy_score(y_test, y_est)
print('正答率(Accuracy) = {:.3f}%'.format(100 * accuracy))

# Precision, recall, f1_score 
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_est)

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
df_kickstar.columns
%%time
y = df_kickstar['state'].values
X = df_kickstar.drop('state', axis=1).values

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, test_size=0.8, random_state=0)

C = 0.01
kernel = "linear"
gamma = 5
clf = SVC(C=C, kernel=kernel, gamma=gamma)
clf.fit(X_train_svm, y_train_svm)
# ラベルを予測
y_est_svm = clf.predict(X_test_svm)

# 対数尤度を表示
print('対数尤度 = {:.3f}'.format(- log_loss(y_test_svm, y_est_svm)))

# 正答率を表示
print('正答率 = {:.3f}%'.format(100 * accuracy_score(y_test_svm, y_est_svm)))
# 正答率を計算
accuracy_svm = accuracy_score(y_test_svm, y_est_svm)
print('正答率(Accuracy) = {:.3f}%'.format(100 * accuracy))

# Precision, recall, f1_score 
precision_svm, recall_svm, f1_score_svm, _ = precision_recall_fscore_support(y_test, y_est)

print('適合率（Precision） = {:.3f}%'.format(100 * precision_svm[0]))
print('再現率（Recall） = {:.3f}%'.format(100 * recall_svm[0]))
print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score_svm[0]))
df_kickstar['name'].value_counts()


