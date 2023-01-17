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
#データの読み込み~表示

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)



Dataset = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

Dataset.head(50)
#データの情報の表示

print(Dataset.shape)

print(Dataset.columns)

print(Dataset.dtypes)
#データの統計量の表示

Dataset.describe(include='all')
#category main_category currency state のcolumnの中に入っている情報の表示

print(Dataset['category'].unique())

print(Dataset['main_category'].unique())

print(Dataset['currency'].unique())

print(Dataset['state'].unique())
#欠損値の確認

print(Dataset.isnull().sum(axis=0))
#説明変数に必要そうなデータの取り出し

df_X = Dataset[['main_category', 'currency', 'deadline', 'goal', 'launched', 'country']]

df_X.head()
#main_category currency country のデータをワンホットエンコードに書き換える

import copy

df_en = copy.deepcopy(df_X)



onehot_main_category = pd.get_dummies(df_en['main_category'])

onehot_currency = pd.get_dummies(df_en['currency'])

onehot_country = pd.get_dummies(df_en['country'])



display(onehot_main_category.head())

display(onehot_currency.head())

display(onehot_country.head())
#deadline launched の日時データを計算に使えるように年・月・日のデータとして取り出す

datetime_deadline = pd.to_datetime(df_X['deadline'])

display(datetime_deadline.head())

print(datetime_deadline.dtype)



datetime_launched = pd.to_datetime(df_X['launched'])

display(datetime_launched.head())

print(datetime_launched.dtype)



datetime_deadline_year = datetime_deadline.dt.year

datetime_deadline_month = datetime_deadline.dt.month

datetime_deadline_day = datetime_deadline.dt.day



datetime_launched_year = datetime_launched.dt.year

datetime_launched_month = datetime_launched.dt.month

datetime_launched_day = datetime_launched.dt.day
#説明変数の設定

X = pd.concat([onehot_main_category, onehot_currency, onehot_country, Dataset['goal'],

               datetime_deadline_year,datetime_deadline_month,datetime_deadline_day, datetime_launched_year, datetime_launched_month, datetime_launched_day],axis=1)

X.head
#目的変数の設定(successful:1 それ以外：0)

y = Dataset['state'].map({'successful':1, 'failed':0, 'canceled':0, 'live':0, 'undefined':0, 'suspended':0})

y.head()
#分類問題

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
#ロジスティクス回帰

lr = LogisticRegression()

lr.fit(X,y)
#モデルの評価

y_pred = lr.predict(X)



accuracy = accuracy_score(y,y_pred)

print('正答率 = {:.3f}%'.format(100*accuracy))



precision, recall, f1_score, _ = precision_recall_fscore_support(y,y_pred)



print('適合率 = {:.3f}%'.format(100*precision[0]))

print('再現率 = {:.3f}%'.format(100*recall[0]))

print('F1値 = {:.3f}%'.format(100*f1_score[0]))