%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数

from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数



import seaborn as sns



import datetime # 日付処理用
df_projects_all = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv",header=0)
# データを出力

display(df_projects_all.head(20))

df_projects_all.describe()





print("state contest")

print(df_projects_all['state'].unique())

print(df_projects_all['state'].value_counts())
# 相関係数を確認

df_projects_all.corr()
# 相関係数をヒートマップにして可視化

sns.heatmap(df_projects_all.corr())

plt.show()
# 要素の取り出し

x = df_projects_all[['goal', 'usd_goal_real']].values

# goalとusd_goal_realは同じ意味なので不適



print(x)



# グラフの確認

df_projects_all.plot.scatter(x='goal', y='usd_goal_real')
# 目的変数

y = df_projects_all['state']



print(y)



# failedとsuccessfulを数値化

y = y.map({'failed':0, 'successful':1, 'canceled':0, 'live':0, 'undefined':0, 'suspended':0})

print(y)



y = y.values
clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

clf.fit(x, y)



# 重みを取得して表示

w0 = clf.intercept_[0]

w1 = clf.coef_[0, 0]

#w2 = clf.coef_[0, 1]

#w3 = clf.coef_[0, 2]

#w4 = clf.coef_[0, 3]

#w5 = clf.coef_[0, 4]

#print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}'.format(w0, w1, w2, w3, w4, w5))

print('w0 = {:.3f}, w1 = {:.3f}'.format(w0, w1))
# ラベルを予測

y_est = clf.predict(x)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(y, y_est)))



# 正答率を表示

print('正答率 = {:.3f}%'.format(100 * accuracy_score(y, y_est)))
# ラベルを予測

y_pred = clf.predict(x)



# 正答率を計算

accuracy =  accuracy_score(y, y_pred)

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))



# Precision, Recall, F1-scoreを計算

precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_pred)



# カテゴリ「successful」に関するPrecision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y, y_est), 

                        index=['正解 = successful', '正解 = failed and other'], 

                        columns=['予測 = failed and other', '予測 = successful'])

conf_mat