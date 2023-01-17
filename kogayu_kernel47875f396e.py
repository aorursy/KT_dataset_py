%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from sklearn.metrics import mean_squared_error, mean_absolute_error # 回帰問題における性能評価に関する関数

from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数

from sklearn.model_selection import KFold # 交差検証法に関する関数



import seaborn as sns

import category_encoders as ce



import datetime # 日付処理用

#df_projects_all = pd.read_csv("./data/ks-projects-201801.csv",header=0)

df_projects_all = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv",header=0)
# データを出力

display(df_projects_all.head(20))

df_projects_all.describe()





# 【変更】stateがfailedとsuccessfulの物のみを使用するように変更

df_projects_all['state'] = df_projects_all['state'].map({'failed':0, 'successful':1})





# 欠損値がある行を削除（対象はstateとusd_goal_real）【新規】

df_projects_all = df_projects_all.dropna(subset=['state', 'usd_goal_real'], how='any')



# データの標準化【新規】

df_projects_all['usd_goal_real'] = (df_projects_all['usd_goal_real'] - df_projects_all['usd_goal_real'].mean()) / df_projects_all['usd_goal_real'].std()

display(df_projects_all.head(20))

df_projects_all.describe()



# データの確認

print("state contest")

print(df_projects_all['state'].unique())

print(df_projects_all['state'].value_counts())



print("main_category contest")

print(df_projects_all['main_category'].unique())

print(df_projects_all['main_category'].value_counts())



print("category contest")

print(df_projects_all['category'].unique())

print(df_projects_all['category'].value_counts())

# 相関係数を確認

df_projects_all.corr()
# 相関係数をヒートマップにして可視化

sns.heatmap(df_projects_all.corr())

plt.show()
# 散布図行列で可視化【新規追加】

pd.plotting.scatter_matrix(df_projects_all, figsize=(10,10))

plt.show()



# usd_goal_realのヒストグラムを表示を行いたかったが断念
# 要素の取り出し【要素の変更】

df_projects_select = df_projects_all[['usd_goal_real','main_category','country']]





# データの標準化（こちらでやるとなぜかwarning

#df_projects_select['usd_goal_real'] = (df_projects_all['usd_goal_real'] - df_projects_all['usd_goal_real'].mean())





# OneHotEncoding【新規】

list_cols = ['main_category', 'country']

# OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。

ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='impute')

df_project_select_oh = ce_ohe.fit_transform(df_projects_select)

df_project_select_oh.head()



# 欠損値がある行を削除（対象はstateとusd_goal_real）【新規】

df_project_select_oh = df_project_select_oh.dropna(how='any')



df_project_select_oh.describe()

x =  df_project_select_oh.values

print(x)
# 目的変数

y = df_projects_all['state']



print(y)



y = y.values
clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

clf.fit(x, y)



# 重みを取得して表示

w0 = clf.intercept_[0]

w1 = clf.coef_[0, 0]

w2 = clf.coef_[0, 1]

w3 = clf.coef_[0, 2]

w4 = clf.coef_[0, 3]

w5 = clf.coef_[0, 4]

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}'.format(w0, w1, w2, w3, w4, w5))

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
print(x.shape)

print(y.shape)
#x = x.reshape(-1,1) # scikit-learnに入力するために整形

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_mae = 0

split_num = 1



kf = KFold(n_splits=n_split, shuffle=True, random_state=1234)

print(kf) 



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, shuffle=True, random_state=1234).split(x, y):

    X_train, y_train = x[train_idx], y[train_idx] #学習用データ

    X_test, y_test = x[test_idx], y[test_idx]     

    

    # 学習用データを使って線形回帰モデルを学習

    clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

    clf.fit(X_train, y_train)



    # テストデータに対する予測を実行

    y_pred_test = clf.predict(X_test)

    

    # テストデータに対するMAEを計算

    mae = mean_absolute_error(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print("MAE = %s"%round(mae, 3))

    print()

    

    cross_valid_mae += mae #後で平均を取るためにMAEを加算

    split_num += 1

    

# MAEの平均値を最終的な汎化誤差値とする

final_mae = cross_valid_mae / n_split

print("Cross Validation MAE = %s"%round(final_mae, 3))