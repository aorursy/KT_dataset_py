%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from sklearn.model_selection import KFold # 交差検証法に関する関数

from  sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix # 回帰問題における性能評価に関する関数

from sklearn.feature_selection import RFECV

import seaborn as sns
df_cloudfound = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# df_cloudfound['state'] = df_cloudfound['state'] == "successful"



display(df_cloudfound.head(10))

df_cloudfound.describe()
# 散布図行列を書いてみる

df_cloudfound = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

#pd.plotting.scatter_matrix(df_cloudfound_sct, figsize=(10,10))
plt.show()# 相関係数を確認

df_cloudfound.corr()
# 相関係数をヒートマップにして可視化

sns.heatmap(df_cloudfound.corr())

plt.show()
# categoryごとのstateの出現頻度を確認

# データ内のcategoryを抽出しcategoryに格納

category=df_cloudfound.groupby('category')

# stateを相対的な頻度に変換

category=category['state'].value_counts(normalize=True).unstack() 

# successfulの降順ソート

category=category.sort_values(by=['successful'],ascending=False)

# 縦棒グラフ（積み上げ）でグラフ作成

category[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
# categoryごとのstateの出現頻度を確認

# データ内のcategoryを抽出しcategoryに格納

main_category=df_cloudfound.groupby('main_category')

# stateを相対的な頻度に変換

main_category=main_category['state'].value_counts(normalize=True).unstack() 

# successfulの降順ソート

main_category=main_category.sort_values(by=['successful'],ascending=False)

# 縦棒グラフ（積み上げ）でグラフ作成

main_category[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
# countryごとのstateの出現頻度を確認

country=df_cloudfound.groupby('country')

country=country['state'].value_counts(normalize=True).unstack()

country=country.sort_values(by=['successful'],ascending=False)

ax=country[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
# currency毎のstateの出現頻度を確認

currency = df_cloudfound.groupby('currency')

currency = currency['state'].value_counts(normalize=True).unstack()

currency = currency.sort_values(by=['successful'],ascending=False)

ax = currency[['successful','failed','canceled','live','suspended','undefined']].plot(kind='bar',stacked=True,figsize=(20,20))
# launched,deadlineからtermを計算

from datetime import datetime

launched = pd.to_datetime(df_cloudfound['launched'])

deadline = pd.to_datetime(df_cloudfound['deadline'])

term = deadline - launched

term = term.dt.total_seconds()
df_cloudfound = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")[['state', 'main_category', 'currency', 'usd_goal_real']]

# df_cloudfound['state'] = df_cloudfound['state'].replace('failed')

df_cloudfoundN = df_cloudfound.query('state == "successful" or state == "failed"')

df_cloudfoundN['term'] = term



# データ表示

display(df_cloudfoundN.head(50))
#df_cloudfound = pd.read_csv("../1_data/ks-projects-201801.csv")[['state', 'main_category', 'currency', 'usd_goal_real']]

# from sklearn.preprocessing import LabelBinarizer



df_cloudfoundN['state'] = df_cloudfoundN['state'] == "successful"#bool型に変換

df_cloudfoundN['state'] = df_cloudfoundN['state'] * 1 #bool型を0,1に変換



#'usd_goal_real'を標準化

df0 = df_cloudfoundN['usd_goal_real']  

df_cloudfoundN['usd_goal_real'] = (df0 - df0.mean()) / (df0.std()) 



#'term'を標準化

df1 = df_cloudfoundN['term']  

df_cloudfoundN['term'] = (df1 - df1.mean()) / (df1.std()) 



#'main_category,currency'をラベルデータを0,1のダミー変数で置き換え&先頭行削除

df_cloudfoundN = pd.get_dummies(df_cloudfoundN, drop_first=True) 

# lb = LabelBinarizer()

# df_cloudfoundN = lb.fit_transform(df_cloudfoundN)





# データ表示

display(df_cloudfoundN.head(50))
# estimatorにモデルをセット

# 今回は回帰問題であるためLinearRegressionを使用

# estimator = LinearRegression(normalize=False)

estimator = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)



# RFECVは交差検証によってステップワイズ法による特徴選択を行う

# cvにはFold（=グループ）の数，scoringには評価指標を指定する

# 今回は分類なのでaccuracyを評価指標に指定

rfecv = RFECV(estimator, cv=10, scoring='accuracy')
train_label = df_cloudfoundN["state"]

train_data = df_cloudfoundN.drop("state", axis=1)



y = train_label.values

X = train_data.values



# fitで特徴選択を実行

rfecv.fit(X, y)
# 特徴のランキングを表示（1が最も重要な特徴）

print('Feature ranking: \n{}'.format(rfecv.ranking_))
# 特徴数とスコアの変化をプロット

# 負のMAEが評価基準になっており，値がゼロに近いほど汎化誤差は小さい

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
# rfecv.support_でランキング1位以外はFalseとするindexを取得できる

# Trueになっている特徴を使用すれば汎化誤差は最小となる

rfecv.support_
# bool型の配列に ~ をつけるとTrueとFalseを反転させることができる

# ここでTrueになっている特徴が削除してもよい特徴

remove_idx = ~rfecv.support_

remove_idx
# 削除してもよい特徴の名前を取得する

remove_feature = train_data.columns[remove_idx]

remove_feature
# drop関数で特徴を削除

df_cloudfoundN = df_cloudfoundN.drop(remove_feature, axis=1)

df_cloudfoundN
y = df_cloudfoundN["state"].values

X = df_cloudfoundN.drop('state', axis=1).values



#X = x.reshape(-1,1) # scikit-learnに入力するために整形

n_split = 5 # グループ数を設定（今回は5分割）



cross_valid_acc = 0

cross_valid_precision = 0

cross_valid_recall = 0

cross_valid_f1_score = 0

split_num = 1



# テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, shuffle=True, random_state=1234).split(X, y):

    X_train, y_train = X[train_idx], y[train_idx] #学習用データ

    X_test, y_test = X[test_idx], y[test_idx]     #テスト用データ

    

    #ロジスティック回帰を実行

    clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234, tol=1e-3)

    clf.fit(X_train, y_train)

    

    # テストデータに対する予測を実行

    # ラベルを予測

    y_pred_test = clf.predict(X_test)



    # 正答率を計算

    accuracy =  accuracy_score(y_test, y_pred_test)

    print("Fold %s"%split_num)

    print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy))



    # Precision, Recall, F1-scoreを計算

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_test)



    # カテゴリ「2000万以上」に関するPrecision, Recall, F1-scoreを表示   

    print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

    print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

    print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))



    cross_valid_acc += accuracy #後で平均を取るためにMAEを加算

    cross_valid_precision += precision #後で平均を取るためにMAEを加算

    cross_valid_recall += recall #後で平均を取るためにMAEを加算

    cross_valid_f1_score += f1_score #後で平均を取るためにMAEを加算

    split_num += 1

    

final_acc =  cross_valid_acc / n_split

final_precision =  cross_valid_precision / n_split

final_recall =  cross_valid_recall / n_split

final_f1_score =  cross_valid_f1_score / n_split

print("Cross Validation")

print('正答率（Accuracy） = {:.3f}%'.format(100 * final_acc))

print('適合率（Precision） = {:.3f}%'.format(100 * final_precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * final_recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * final_f1_score[0]))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred_test), 

                        index=['正解 = Failed', '正解 = Successful'], 

                        columns=['予測 = Failed', '予測 = Successful'])

conf_mat