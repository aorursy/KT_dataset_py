# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
##KICKSTARTER

#説明変数：'category','country','usd_goal_real','period(=launched-deadline)'

#目的変数：state(successfulとfailedのみ　他は削除)

#モデル：ロジスティック回帰モデル

#評価基準：対数尤度

#最適化：確率的勾配法

#前処理：標準化、One-Hotエンコーディング

#手法：ホールドアウト法、交差検証法
#準備

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import SGDClassifier,Ridge,Lasso,ElasticNet

from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler

from sklearn.model_selection import train_test_split,KFold

np.random.seed(1234)
#データセットの読み込み

df_ks = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

display(df_ks.head(10))

display(df_ks.describe())

display(df_ks.isnull().sum())#欠損値

display(df_ks.dtypes)
#データ検証
#currencyとcountryの関係

df_currency_country = df_ks.groupby('country')

df_currency_country = df_currency_country['currency'].value_counts(normalize=True).unstack(fill_value=0)

display(df_currency_country)



print('countryの種類と数を調べる')

print(df_ks['country'].value_counts(dropna=False))
#categoryとmain_categoryの関係

df_categories = df_ks.groupby('category')

df_categories = df_categories['main_category'].value_counts(normalize=True).unstack(fill_value=0)

display(df_categories)



#categoryの種類と数

print('categoryの種類と数を調べる')

print(df_ks['category'].value_counts(dropna=False))



#main_categoryの種類と数

print('main_categoryの種類と数を調べる')

print(df_ks['main_category'].value_counts(dropna=False))
#Stateの種類と数

print(df_ks['state'].value_counts(dropna=False))
#category毎にStateとの相関をグラフ化

category_ = df_ks.groupby('category')

category_ = category_['state'].value_counts(normalize=True).unstack()

category_ = category_.sort_values(by=['successful'],ascending=True)

category_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,30))
#country毎にStateとの相関をグラフ化

country_ = df_ks.groupby('country')

country_ = country_['state'].value_counts(normalize=True).unstack()

country_ = country_.sort_values(by=['successful'],ascending=True)

country_[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh', stacked=True,figsize=(13,7))
#データ整理
#欠損値削除

df_ks = df_ks.dropna()

#display(df_ks.isnull().sum())



#日付をdatetime型に変換し、'launched'から'deadline'までの日数'period'を算出

df_ks['deadline'] = pd.to_datetime(df_ks['deadline'], errors = 'coerce')

df_ks['launched'] = pd.to_datetime(df_ks['launched'], errors = 'coerce')

df_ks['period'] = (df_ks['deadline'] - df_ks['launched']).dt.days

#display(df_ks.head(10))



#stateがsuccessfulとfailedのデータのみ抽出

df_ks_S = df_ks[df_ks['state']=='successful']

df_ks_F = df_ks[df_ks['state']=='failed']

df_ks2 = pd.concat([df_ks_S,df_ks_F])



#カテゴリ変数を数値データに変換

le =LabelEncoder()

le = le.fit(df_ks2['state'])

df_ks2['state'] = le.transform(df_ks2['state'])

#display(df_ks2.head(10))



#不要変数を削除

df_ks3 = df_ks2.drop(['ID','name','deadline','launched','currency','backers','pledged','usd pledged','main_category','goal','usd_pledged_real'], axis=1)

#display(df_ks3.head(10))



#並び替え

df_ks4 = df_ks3.iloc[:,[1,0,2,3,4]]

#display(df_ks4.head(10))



#categoryとcountryをダミー変換

df_dummy_ca = pd.get_dummies(df_ks4['category'])

df_dummy_co = pd.get_dummies(df_ks4['country'])

df_ks5 = pd.concat([df_ks4.drop(['category'],axis=1),df_dummy_ca],axis=1)

df_ks6 = pd.concat([df_ks5.drop(['country'],axis=1),df_dummy_co],axis=1)

display(df_ks6.head(10))
#学習・予測
#ホールドアウト法

y = df_ks6['state'].values #'state' の値をyに代入する

X = df_ks6.drop('state', axis=1).values #'state'以外の変数をXに代入する

test_size = 0.3 #テストデータの割合を決める

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)



#標準化

std = StandardScaler()

X_train_std = std.fit_transform(X_train) #訓練データの標準化

X_test_std = std.transform(X_test) #テストデータの標準化※fit_transformはダメ



#学習の実行

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,tol=1e-3)

#loss：損失関数　max_iter：学習の最大回数　fit_intercept：切片を求める

clf_std = clf.fit(X_train_std, y_train)



#結果の表示

y_prd_test_std = clf_std.predict(X_test_std)

cm = pd.DataFrame(confusion_matrix(y_test, y_prd_test_std,labels=[1,0]),index=['正解=成功','正解=失敗'],columns=['予測=成功','予測=失敗'])#1=positive(successful),0=negative(failed)

print(cm) #混同行列

print('対数尤度 = {:.3f}'.format(-log_loss(y_test,y_prd_test_std))) #対数尤度を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test_std))) #正答率を表示

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test_std) #適合率・再現率・F1値を計算

print('適合率（Precision） = {:.3f}%'.format(100 * precision[1])) #適合率を表示

print('再現率（Recall） = {:.3f}%'.format(100 * recall[1])) #再現率を表示

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1])) #F1値を表示
#交差検証法

n_split = 5 #分割数を設定



cross_valid_acc = 0

cross_valid_precision = 0

cross_valid_recall = 0

cross_valid_f1_score = 0

split_num = 1



#テスト役を交代させながら学習と評価を繰り返す

for train_idx, test_idx in KFold(n_splits=n_split, shuffle=True, random_state=1234).split(X, y):

    X_train, y_train = X[train_idx], y[train_idx] #訓練データ

    X_test, y_test = X[test_idx], y[test_idx] #テストデータ

    

    #標準化

    std = StandardScaler()

    X_train_std = std.fit_transform(X_train) #訓練データの標準化

    X_test_std = std.transform(X_test) #テストデータの標準化

    

    #学習の実行

    clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,tol=1e-3)

    clf_std = clf.fit(X_train_std, y_train)

    

    #結果の表示

    y_prd_test_std = clf_std.predict(X_test_std)

    cm = confusion_matrix(y_test, y_prd_test_std,labels=[1,0])#1=positive(successful),0=negative(failed)

    print("Fold %s"%split_num)

    print('対数尤度 = {:.3f}'.format(-log_loss(y_test,y_prd_test_std))) #対数尤度を表示

    accuracy = accuracy_score(y_test, y_prd_test_std)

    print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test_std))) #正答率を表示

    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test_std) #適合率・再現率・F1値を計算

    print('適合率（Precision） = {:.3f}%'.format(100 * precision[1])) #適合率を表示

    print('再現率（Recall） = {:.3f}%'.format(100 * recall[1])) #再現率を表示

    print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1])) #F1値を表示



    cross_valid_acc += accuracy

    cross_valid_precision += precision

    cross_valid_recall += recall

    cross_valid_f1_score += f1_score

    split_num += 1

    

final_acc =  cross_valid_acc / n_split

final_precision =  cross_valid_precision / n_split

final_recall =  cross_valid_recall / n_split

final_f1_score =  cross_valid_f1_score / n_split

print("Cross Validation")

print('正答率（Accuracy） = {:.3f}%'.format(100 * final_acc))

print('適合率（Precision） = {:.3f}%'.format(100 * final_precision[1]))

print('再現率（Recall） = {:.3f}%'.format(100 * final_recall[1]))

print('F1値（F1-score） = {:.3f}%'.format(100 * final_f1_score[1]))