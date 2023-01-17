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
!pip install pydotplus
#準備

%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pydotplus

from sklearn.linear_model import SGDClassifier,Ridge,Lasso,ElasticNet

from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler

from sklearn.model_selection import train_test_split,KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO

from IPython.display import Image

np.random.seed(1234)
#データセットの読み込み

df_ks = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

display(df_ks.head(10))

display(df_ks.describe())

display(df_ks.isnull().sum())#欠損値

display(df_ks.dtypes)
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
#欠損値削除

df_ks = df_ks.dropna()

#display(df_ks.isnull().sum())
#日付をdatetime型に変換し、'launched'から'deadline'までの日数'period'を算出

df_ks['deadline'] = pd.to_datetime(df_ks['deadline'], errors = 'coerce')

df_ks['launched'] = pd.to_datetime(df_ks['launched'], errors = 'coerce')

df_ks['period'] = (df_ks['deadline'] - df_ks['launched']).dt.days

df_ks = df_ks[ (df_ks['period'] < 1000)]

df_ks = df_ks[ (df_ks['period'] > 1)]
#periodをlog化

df_ks['log_period']=np.log(df_ks['period'])

sns.distplot(df_ks['log_period']);
#usd_goal_realをlog化

df_ks['log_ugr']=np.log(df_ks['usd_goal_real'])

sns.distplot(df_ks['log_ugr']);
#usd_goal_real/period

df_ks['ugr/period']=np.log(df_ks['usd_goal_real']/df_ks['period'])

df_ks = df_ks.dropna()

display(df_ks.describe())

sns.distplot(df_ks['ugr/period']);
df_ks['launched_year'] = (df_ks['launched']).dt.year

df_ks['launched_month'] = (df_ks['launched']).dt.month

df_ks['launched_day'] = (df_ks['launched']).dt.day

df_ks['deadline_year'] = (df_ks['deadline']).dt.year

df_ks['deadline_month'] = (df_ks['deadline']).dt.month

df_ks['deadline_day'] = (df_ks['deadline']).dt.day

display(df_ks.head())
#period毎にStateとの相関をグラフ化

period_ = df_ks.groupby('period')

period_ = period_['state'].value_counts(normalize=True).unstack()

period_ = period_.sort_values(by=['successful'],ascending=True)

period_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,20))
#launched_year毎にStateとの相関をグラフ化

launched_year_ = df_ks.groupby('launched_year')

launched_year_ = launched_year_['state'].value_counts(normalize=True).unstack()

launched_year_ = launched_year_.sort_values(by=['successful'],ascending=True)

launched_year_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,3))
#launched_month毎にStateとの相関をグラフ化

launched_month_ = df_ks.groupby('launched_month')

launched_month_ = launched_month_['state'].value_counts(normalize=True).unstack()

launched_month_ = launched_month_.sort_values(by=['successful'],ascending=True)

launched_month_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,3))
#launched_day毎にStateとの相関をグラフ化

launched_day_ = df_ks.groupby('launched_day')

launched_day_ = launched_day_['state'].value_counts(normalize=True).unstack()

launched_day_ = launched_day_.sort_values(by=['successful'],ascending=True)

launched_day_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,7))
#deadline_year毎にStateとの相関をグラフ化

deadline_year_ = df_ks.groupby('deadline_year')

deadline_year_ = deadline_year_['state'].value_counts(normalize=True).unstack()

deadline_year_ = deadline_year_.sort_values(by=['successful'],ascending=True)

deadline_year_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,3))
#deadline_month毎にStateとの相関をグラフ化

deadline_month_ = df_ks.groupby('deadline_month')

deadline_month_ = deadline_month_['state'].value_counts(normalize=True).unstack()

deadline_month_ = deadline_month_.sort_values(by=['successful'],ascending=True)

deadline_month_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,3))
#deadline_day毎にStateとの相関をグラフ化

deadline_day_ = df_ks.groupby('deadline_day')

deadline_day_ = deadline_day_['state'].value_counts(normalize=True).unstack()

deadline_day_ = deadline_day_.sort_values(by=['successful'],ascending=True)

deadline_day_[['successful','failed','live','canceled','suspended']].plot(kind='barh', stacked=True,figsize=(13,7))
#stateがsuccessfulとfailedのデータのみ抽出

df_ks_S = df_ks[df_ks['state']=='successful']

df_ks_F = df_ks[df_ks['state']=='failed']

df_ks2 = pd.concat([df_ks_S,df_ks_F])

display(df_ks2.head(10))
#不要変数を削除

df_ks3 = df_ks2.drop(['ID','name','deadline','launched','currency','backers','pledged','usd pledged','main_category','goal','usd_pledged_real','usd_goal_real','period'], axis=1)

display(df_ks3.head(10))
#カテゴリ変数を数値データに変換

le =LabelEncoder()

df_ks3['state'] = le.fit_transform(df_ks3['state'])

#df_ks3['country'] = le.fit_transform(df_ks3['country'])

#df_ks3['category'] = le.fit_transform(df_ks3['category'])

df_dummy_ca = pd.get_dummies(df_ks3['category'])

df_dummy_co = pd.get_dummies(df_ks3['country'])

df_ks4 = pd.concat([df_ks3.drop(['category'],axis=1),df_dummy_ca],axis=1)

df_ks5 = pd.concat([df_ks4.drop(['country'],axis=1),df_dummy_co],axis=1)

display(df_ks5.head(10))

display(df_ks5.dtypes)
#並び替え

#df_ks4 = df_ks3.iloc[:,[1,0,2,3,4,5,6,7,8,9,10]]

#display(df_ks4.head(10))
#ホールドアウト法

y = df_ks5['state'].values #'state' の値をyに代入する

X = df_ks5.drop('state', axis=1).values #'state'以外の変数をXに代入する

test_size = 0.3 #テストデータの割合を決める

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)
#ロジスティック回帰



#標準化

std = StandardScaler()

X_train_std = std.fit_transform(X_train) #訓練データの標準化

X_test_std = std.transform(X_test) #テストデータの標準化※fit_transformはダメ



#学習の実行

sgd = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,tol=1e-3)

#loss：損失関数　max_iter：学習の最大回数　fit_intercept：切片を求める

sgd_std = sgd.fit(X_train_std, y_train)



#結果の表示

y_prd_test_std = sgd_std.predict(X_test_std)

cm = pd.DataFrame(confusion_matrix(y_test, y_prd_test_std,labels=[1,0]),index=['正解=成功','正解=失敗'],columns=['予測=成功','予測=失敗'])#1=positive(successful),0=negative(failed)

display(cm) #混同行列

print('対数尤度 = {:.3f}'.format(-log_loss(y_test,y_prd_test_std))) #対数尤度を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test_std))) #正答率を表示

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test_std) #適合率・再現率・F1値を計算

print('適合率（Precision） = {:.3f}%'.format(100 * precision[1])) #適合率を表示

print('再現率（Recall） = {:.3f}%'.format(100 * recall[1])) #再現率を表示

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1])) #F1値を表示
#ランダムフォレスト



#学習の実行

rf = RandomForestClassifier(n_estimators=10, 

                            max_depth=10, 

                            criterion="gini",

                            min_samples_leaf=2, 

                            min_samples_split=2, 

                            random_state=1234, 

                            n_jobs=-1)

rf = rf.fit(X_train, y_train)



#結果の表示

y_prd_test = rf.predict(X_test)

cm = pd.DataFrame(confusion_matrix(y_test, y_prd_test,labels=[1,0]),index=['正解=成功','正解=失敗'],columns=['予測=成功','予測=失敗'])#1=positive(successful),0=negative(failed)

display(cm) #混同行列

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test))) #正答率を表示

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test) #適合率・再現率・F1値を計算

print('適合率（Precision） = {:.3f}%'.format(100 * precision[1])) #適合率を表示

print('再現率（Recall） = {:.3f}%'.format(100 * recall[1])) #再現率を表示

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1])) #F1値を表示
#決定木の描画



feature_names = df_ks5.drop('state', axis=1).columns

class_names = df_ks2['state'].values



for i, est in enumerate(rf.estimators_):

    print(i)

    

    # 決定木の描画

    dot_data = StringIO() #dotファイル情報の格納先

    export_graphviz(est, out_file=dot_data,  

                         feature_names=feature_names,  

                         class_names=class_names,  

                         filled=True, rounded=True,  

                         special_characters=False) 

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 

    display(Image(graph.create_png()))
#決定木



#学習の実行

dt = RandomForestClassifier(n_estimators=1, 

                            max_depth=10, 

                            criterion="gini",

                            min_samples_leaf=2, 

                            min_samples_split=2, 

                            random_state=1234, 

                            n_jobs=-1)

dt = dt.fit(X_train, y_train)



#結果の表示

y_prd_test = dt.predict(X_test)

cm = pd.DataFrame(confusion_matrix(y_test, y_prd_test,labels=[1,0]),index=['正解=成功','正解=失敗'],columns=['予測=成功','予測=失敗'])#1=positive(successful),0=negative(failed)

display(cm) #混同行列

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test))) #正答率を表示

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test) #適合率・再現率・F1値を計算

print('適合率（Precision） = {:.3f}%'.format(100 * precision[1])) #適合率を表示

print('再現率（Recall） = {:.3f}%'.format(100 * recall[1])) #再現率を表示

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1])) #F1値を表示
#決定木の描画



feature_names2 = df_ks5.drop('state', axis=1).columns

class_names2 = df_ks2['state'].values



for i, est in enumerate(dt.estimators_):

    print(i)

    

    # 決定木の描画

    dot_data2 = StringIO() #dotファイル情報の格納先

    export_graphviz(est, out_file=dot_data2,  

                         feature_names=feature_names2,  

                         class_names=class_names2,  

                         filled=True, rounded=True,  

                         special_characters=False) 

    graph = pydotplus.graph_from_dot_data(dot_data2.getvalue()) 

    display(Image(graph.create_png()))