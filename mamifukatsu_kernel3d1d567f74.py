import datetime as dt

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix 
# データの読み込み

import codecs

with codecs.open('../input/ks-projects-201612.csv', "r", "Shift-JIS", "ignore") as file:

    df_crdf = pd.read_csv(file, delimiter=",")

df_crdf.columns = df_crdf.columns.str.replace(" ", "")
df_crdf.head()
# 空白の列を削除

df_crdf = df_crdf.drop(['Unnamed:13','Unnamed:14','Unnamed:15','Unnamed:16'], axis=1)
# データ情報を確認

df_crdf.info()
# state列のsuccessfulをTrueにそれ以外をFalseに変換

df_crdf['state'] = df_crdf['state'] == 'successful'
# 日時の項目をdatetimeに変更

df_crdf['deadline'] = pd.to_datetime(df_crdf['deadline'], errors = 'coerce')

df_crdf['launched'] = pd.to_datetime(df_crdf['launched'], errors = 'coerce')
# 数値の項目をfloatに変更

df_crdf['goal'] = pd.to_numeric(df_crdf['goal'], errors ='coerce')

df_crdf['pledged'] = pd.to_numeric(df_crdf['pledged'], errors ='coerce')

df_crdf['backers'] = pd.to_numeric(df_crdf['backers'], errors ='coerce')

df_crdf['usdpledged'] = pd.to_numeric(df_crdf['usdpledged'], errors ='coerce')
# launchedとdeadlineの間の日数を示すperiod列を作成

df_crdf['period'] = df_crdf['deadline'] - df_crdf['launched']
# period内の日数のみをdays列に抽出

days = []

for i in range(len(df_crdf['period'])):

    days.append(df_crdf['period'][i].days)



df_crdf['days'] = days
# 不要になった日時の項目を削除

df_crdf = df_crdf.drop(['deadline', 'launched', 'period'], axis=1)
# 上手くデータ処理できなかった項目は今回は削除

df_crdf = df_crdf.drop(['ID','name','category','main_category','currency','country'], axis=1)
# 各データ型を再確認

df_crdf.dtypes
# 欠損値を確認

df_crdf.isnull().sum()
# 欠損値を含む行を削除

df_crdf = df_crdf.dropna()
# 欠損値を再確認

df_crdf.isnull().sum()
# 要素数の再確認

df_crdf.shape
#各データの相関係数

df_crdf.corr()
# 重みの学習

y = df_crdf['state'].values

X = df_crdf.drop('state', axis=1).values

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True)

clf.fit(X, y)
# 重みを変数に代入

w0 = clf.intercept_[0]

w1 = clf.coef_[0, 0]

w2 = clf.coef_[0, 1]

w3 = clf.coef_[0, 2]

w4 = clf.coef_[0, 3]

w5 = clf.coef_[0, 4]

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}'.format(w0, w1, w2, w3, w4, w5))
# ラベルを予測

y_prd = clf.predict(X)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(y, y_prd)))



# 正答率を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y, y_prd)))



# Precision, Recall, F1-scoreを計算

precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_prd)



# カテゴリ「2000万以上」に関するPrecision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
# 予測値と正解のクロス集計

conf_mat = pd.DataFrame(confusion_matrix(y, y_prd), 

                        index=['正解 = not successful', '正解 = successful'], 

                        columns=['予測 = not successful', '予測 = successful'])

conf_mat