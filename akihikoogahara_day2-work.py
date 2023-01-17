# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df_data=pd.read_csv('../input/ks-projects-201801.csv')
df_data.head(10) 
df_data.describe() # 要約統計量の表示
pd.plotting.scatter_matrix(df_data,figsize=(20,20)) # 散布図行列を計算、表示

plt.show()
df_data.corr() # 各相関係数を表示
sns.heatmap(df_data.corr()) # 各相関係数のヒートマップを表示

plt.show()
# categoryの種類と数について

print(df_data['category'].value_counts(dropna=False)) # データ内の[category]をインデックス指定して種類と数量を調べる
# main categoryについても種類と数量を調べる

print(df_data['main_category'].value_counts(dropna=False))
# goalとusd_goal_realの散布図行列を確認

pd.plotting.scatter_matrix(df_data[['goal','usd_goal_real']],figsize=(10,10))

plt.show()



#相関係数を確認

corr1=df_data[['goal','usd_goal_real']].corr()

print(corr1)
# stateの種類と数量を確認

print(df_data['state'].value_counts(dropna=False))
# category毎のstateを確認する

category=df_data.groupby('category')

category=category['state'].value_counts(normalize=True).unstack()

category=category.sort_values(by=['successful'],ascending=True)

category[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh',stacked=True,figsize=(20,30))
# deadline毎のstateを確認

df_data_deadline=df_data.copy()

df_data_deadline['deadline_YM']=df_data_deadline['deadline'].apply(lambda x: x[0:7])

deadline=df_data_deadline.groupby('deadline_YM')

deadline=deadline['state'].value_counts(normalize=True).unstack()

ax=deadline[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh',stacked=True,figsize=(13,30))

plt.legend(loc='upper right')
# launched毎のstateを確認

df_data_launched=df_data.copy()

df_data_launched['launched_YM']=df_data_launched['launched'].apply(lambda x: x[0:7])

launched=df_data_launched.groupby('launched_YM')

launched=launched['state'].value_counts(normalize=True).unstack()

ax=launched[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh',stacked=True,figsize=(13,30))

plt.legend(loc='upper right')
# goal毎のstateを確認する

df_data_goal=df_data.copy()

df_data_goal['goal']=df_data_goal['goal'].apply(lambda x: round(x/100000))

goal=df_data_goal.groupby('goal')

goal=goal['state'].value_counts(normalize=True).unstack()

ax=goal[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh',stacked=True,figsize=(13,30))

plt.legend(loc='upper right')
# country毎のstateを確認する

df_data_country=df_data.copy()

df_data_country['country']=df_data_country['country'].apply(lambda x : x[0:7])

country=df_data_country.groupby('country')

country=country['state'].value_counts(normalize=True).unstack()

country=country.sort_values(by=['successful'],ascending=True)

ax=country[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh',stacked=True,figsize=(13,30))

plt.legend(loc='upper right')
df_data_currency=df_data.copy()

df_data_currency['currency']=df_data_currency['currency'].apply(lambda x : x[0:7])

currency=df_data_currency.groupby('currency')

currency=currency['state'].value_counts(normalize=True).unstack()

currency=currency.sort_values(by=['successful'],ascending=True)

ax=currency[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh',stacked=True,figsize=(13,30))

plt.legend(loc='upper right')
from datetime import datetime
df_data_term=df_data.copy()

df_data_term['term']=pd.to_datetime(df_data_term['deadline']).map(pd.Timestamp.timestamp) - pd.to_datetime(df_data_term['launched'].apply(lambda x: x[0:10])).map(pd.Timestamp.timestamp)

term=df_data_term.groupby('term')

term=term['state'].value_counts(normalize=True).unstack()

ax=term[['successful','failed','live','canceled','suspended','undefined']].plot(kind='barh',stacked=True,figsize=(15,30))

plt.legend(loc='upper left')
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss,accuracy_score,confusion_matrix
# Scikit-learnによるロジスティック回帰実装(説明変数 5つ)

y = df_data["〇〇〇"].values

X = df_data[['category', 'currency','goal','country','term']].values

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True, random_state=1234)

clf.fit(X, y)



# 重みを取得して表示

w0 = clf.intercept_[0]

w1 = clf.coef_[0,0,0,0,0]

w2 = clf.coef_[0,1,0,0,0]

w3 = clf.coef_[0,0,1,0,0]

w4 = clf.coef_[0,0,0,1,0]

w5 = clf.coef_[0,0,0,0,1]

print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f},w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}'.format(w0,w1,w2,w3,w4,w5))