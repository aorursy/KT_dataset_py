%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数

import seaborn as sns
df_cloudfound = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

df_cloudfound['state'] = df_cloudfound['state'] == "successful"



# データ表示

display(df_cloudfound.head(10))

df_cloudfound.describe()
# 散布図行列を書いてみる

df_cloudfound_sct = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

pd.plotting.scatter_matrix(df_cloudfound_sct, figsize=(10,10))
plt.show()# 相関係数を確認

df_cloudfound.corr()
# 相関係数をヒートマップにして可視化

sns.heatmap(df_cloudfound.corr())

plt.show()
# categoryごとのstateの出現頻度を確認

# データ内のcategoryを抽出しcategoryに格納

category=df_cloudfound_sct.groupby('category')

# stateを相対的な頻度に変換

category=category['state'].value_counts(normalize=True).unstack() 

# successfulの降順ソート

category=category.sort_values(by=['successful'],ascending=False)

# 縦棒グラフ（積み上げ）でグラフ作成

category[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
# countryごとのstateの出現頻度を確認

country=df_cloudfound_sct.groupby('country')

country=country['state'].value_counts(normalize=True).unstack()

country=country.sort_values(by=['successful'],ascending=False)

ax=country[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
# currency毎のstateの出現頻度を確認

currency = df_cloudfound_sct.groupby('currency')

currency = currency['state'].value_counts(normalize=True).unstack()

currency = currency.sort_values(by=['successful'],ascending=False)

ax = currency[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
df_cloudfound = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")[['state', 'main_category', 'currency', 'country', 'goal']]

df_cloudfound['state'] = df_cloudfound['state'] == "successful"#bool型に変換

df_cloudfound['state'] = df_cloudfound['state'] * 1 #bool型を0,1に変換

#'goal'を0～1の範囲に正規化

df0 = df_cloudfound['goal']  

df_cloudfound['goal'] = (df0 - df0.min()) / (df0.max() - df0.min())

#'main_category'などラベルデータを0,1のダミー変数で置き換え&先頭行削除

df_cloudfound = pd.get_dummies(df_cloudfound, drop_first=True) 





# データ表示

display(df_cloudfound.head())
y = df_cloudfound["state"].values

X = df_cloudfound.drop('state', axis=1).values





clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

clf.fit(X, y)



# 重みを取得して表示

# w0 = clf.intercept_[0]

# w1 = clf.coef_[0, 0]

# w2 = clf.coef_[0, 1]

# w3 = clf.coef_[0, 2]

# w4 = clf.coef_[0, 3]

# w5 = clf.coef_[0, 4]





# print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}'.format(w0, w1, w2, w3, w4, w5))

print('回帰係数')

print(clf.coef_)
# ラベルを予測

y_pred = clf.predict(X)



# 正答率を計算

accuracy =  accuracy_score(y, y_pred)

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))



# Precision, Recall, F1-scoreを計算

precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_pred)



# カテゴリ「2000万以上」に関するPrecision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y, y_pred), 

                        index=['正解 = Failed', '正解 = Successful'], 

                        columns=['予測 = Failed', '予測 = Successful'])

conf_mat