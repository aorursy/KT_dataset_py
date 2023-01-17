import datetime as dt

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix 

from sklearn.metrics import mean_absolute_error
# データの読み込み

import codecs

with codecs.open('../input/ks-projects-201612.csv', "r", "Shift-JIS", "ignore") as file:

    df_crdf = pd.read_csv(file, delimiter=",")

df_crdf.columns = df_crdf.columns.str.replace(" ", "")
df_crdf.head()
# 空白の列を削除

df_crdf = df_crdf.drop(['Unnamed:13', 'Unnamed:14', 'Unnamed:15', 'Unnamed:16',], axis=1)
# 立上げ時には知り得ない情報を削除

df_crdf = df_crdf.drop(['pledged', 'usdpledged', 'backers'], axis=1)
# データ情報を確認

df_crdf.info()
# 数値の項目をfloatに変更

df_crdf['goal'] = pd.to_numeric(df_crdf['goal'], errors ='coerce')
# 日時の項目をdatetimeに変更

df_crdf['deadline'] = pd.to_datetime(df_crdf['deadline'], errors = 'coerce')

df_crdf['launched'] = pd.to_datetime(df_crdf['launched'], errors = 'coerce')
# launchedとdeadlineの間の日数を示すperiod列を作成

df_crdf['period'] = (df_crdf['deadline'] - df_crdf['launched']).dt.days
# 不要なデータは削除

df_crdf = df_crdf.drop(['ID','name','category','deadline','launched'], axis=1)
# 各データ型を再確認

df_crdf.dtypes
#列を見やすいように並び替える

#df_crdf = df_crdf.ix[:,[4,2,8,5,3,7,0,1,6]]

df_crdf = df_crdf.ix[:,[3,2,5,0,1,4]]

df_crdf.head()
# 欠損値を確認

df_crdf.isnull().sum()
# 欠損値を含む行を削除

df_crdf = df_crdf.dropna()
# state列の内容を確認

df_crdf['state'].unique()



# state列の'successful'と'failed'以外を削除

df_crdf = df_crdf[(df_crdf['state'] == 'successful') | (df_crdf['state'] == 'failed')]



# state列のsuccessfulをTrueにそれ以外をFalseに変換

df_crdf['state'] = df_crdf['state'] == 'successful'
# 'main_category'の中身を確認

df_crdf['main_category'].unique()
# 'main_category'をダミー変数にする

df_dummy = pd.get_dummies(df_crdf['main_category'])

df_crdf = pd.concat([df_crdf.drop(['main_category'],axis=1),df_dummy],axis=1)
# 'currency'の中身を確認

df_crdf['currency'].unique()
# 'currency'をダミー変数にする

df_dummy = pd.get_dummies(df_crdf['currency'])

df_crdf = pd.concat([df_crdf.drop(['currency'],axis=1),df_dummy],axis=1)
# 'country'の中身を確認

df_crdf['country'].unique()
# 'country'をダミー変数にする

df_dummy = pd.get_dummies(df_crdf['country'])

df_crdf = pd.concat([df_crdf.drop(['country'],axis=1),df_dummy],axis=1)
df_crdf.head(10)
#各データの相関係数

df_crdf.corr()
# 相関係数をヒートマップにして可視化

sns.heatmap(df_crdf.corr())

plt.show()
#ホールドアウト法

y = df_crdf['state'].values

X = df_crdf.drop('state', axis=1).values



test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)
#データの標準化

stdsc = StandardScaler()

X_train_stand = stdsc.fit_transform(X_train)

X_test_stand = stdsc.transform(X_test)
# 重みの学習

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True)

clf.fit(X_train, y_train)
# 訓練用データに対する予測

y_prd_train = clf.predict(X_train)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(y_train, y_prd_train)))



# 正答率を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_train, y_prd_train)))



# Precision, Recall, F1-scoreを計算

precision, recall, f1_score, _ = precision_recall_fscore_support(y_train, y_prd_train)



# Precision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
# テストデータに対する予測

y_prd_test = clf.predict(X_test)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(y_test, y_prd_test)))



# 正答率を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test)))



# Precision, Recall, F1-scoreを計算

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test)



# カテゴリ「2000万以上」に関するPrecision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
# 訓練データの予測値と正解のクロス集計

conf_mat_train = pd.DataFrame(confusion_matrix(y_train, y_prd_train), 

                        index=['正解 = not successful', '正解 = successful'], 

                        columns=['予測 = not successful', '予測 = successful'])

conf_mat_train
# テストデータの予測値と正解のクロス集計

conf_mat_test = pd.DataFrame(confusion_matrix(y_test, y_prd_test), 

                        index=['正解 = not successful', '正解 = successful'], 

                        columns=['予測 = not successful', '予測 = successful'])

conf_mat_test
#LASSOによる特徴選択

estimator = LassoCV(normalize=True, cv=10)

sfm = SelectFromModel(estimator, threshold=1e-5)
# fitで特徴選択を実行

sfm.fit(X, y)
# 使用する特徴をTrueで表示する

sfm.get_support()
# 削除すべき特徴の名前を取得 

removed_idx  = ~sfm.get_support()

df_crdf.drop('state', axis=1).columns[removed_idx]
# LASSOで得た各特徴の係数の値を確認する

abs_coef = np.abs(sfm.estimator_.coef_)

abs_coef
# 係数を棒グラフで表示

plt.barh(np.arange(0, len(abs_coef)), abs_coef, tick_label=df_crdf.drop('state', axis=1).columns.values)

plt.show()
# 特徴選択後に学習をする

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)



stdsc = StandardScaler()

X_train_stand = stdsc.fit_transform(X_train)

X_test_stand = stdsc.transform(X_test)



clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True)

clf.fit(X_train, y_train)
# 訓練用データに対する予測

y_prd_train = clf.predict(X_train)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(y_train, y_prd_train)))



# 正答率を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_train, y_prd_train)))



# Precision, Recall, F1-scoreを計算

precision, recall, f1_score, _ = precision_recall_fscore_support(y_train, y_prd_train)



# Precision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))
# テストデータに対する予測

y_prd_test = clf.predict(X_test)



# 対数尤度を表示

print('対数尤度 = {:.3f}'.format(- log_loss(y_test, y_prd_test)))



# 正答率を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test)))



# Precision, Recall, F1-scoreを計算

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test)



# カテゴリ「2000万以上」に関するPrecision, Recall, F1-scoreを表示

print('適合率（Precision） = {:.3f}%'.format(100 * precision[0]))

print('再現率（Recall） = {:.3f}%'.format(100 * recall[0]))

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[0]))