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
#df_train,df_testとしてそれぞれ読み込み

df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

df_gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

#df_gender_submission.to_csv('gender_submission.csv', index=False)

#可視化用のモジュールをimport

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
#データの概観をする

#print(df_train.head(5))

#print(df_train.shape)

#print(df_train.describe())

#print(len(df_train))

#print(df_train.columns)

#print(len(df_train.columns))

#print(df_train.isnull().sum())

#print(df_train.isnull().sum(axis=1))

#df.info()で各列の欠損値の数とデータの型がわかる

print(df_train.info())

#変数名['列名'] .列名

#print(df_train['Age'])

#変数名[開始行：終了行]

#print(df_train[1:5])
#訓練データとテストデータを結合する

df_full=pd.concat([df_train,df_test],axis = 0, ignore_index = True)

print(df_full.shape)

print(df_full.describe())

#数値以外のデータも要約統計量を表示

print(df_full.describe(include='all'))
sns.countplot(x='Survived',data=df_train)

plt.title('死亡者と生存者の数')

plt.xticks([0,1],['死亡者','生存者'])

#Survived列の値を集計

df_train['Survived'].value_counts()
#男女別の生存者数を可視化

sns.countplot(x='Survived',hue='Sex',data=df_train)

#tick(s)はしるしの意味

plt.xticks([0.0,1.0],['dead','survived'])

plt.title('dead or survived group by Sex')
#チケットクラス別の生存者数を可視化

#sns.countplot(x='Survived',hue='Pclass',data=df_train)

#plt.xticks([0.01,1.0],['dead','survived'])



#チケットクラス別の生存割合を表示する

#df_train[['Pclass','Survived']].groupby(['Pclass']).mean()



#全体のヒストグラム

#dropna()は欠損値を除外するメソッド

#Kdeはカーネル密度推定のことTrue

sns.distplot(df_train['Age'].dropna(), kde=True,bins =30 ,label = 'all')