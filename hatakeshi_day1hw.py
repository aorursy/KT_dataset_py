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
### KICKSTARTER
##説明変数として使えるデータ（打ち上げ前に分かる情報）

#'ID','name','category','main_category','currency','country','deadline','launched','goal','usd_goal_real'

##説明変数として使えないデータ（打ち上げ後に分かる情報）

#'backers','pledged','usd_pledged','usd_pledged_real'

##目的変数

#課題①state

#課題②usd_pledged_real/usd_goal_real（=upr/ugr）
##課題①Stateを予測

#採用した説明変数：'main_category','country','usd_goal_real','period(=launched-deadline)'

#モデル：ロジスティック回帰モデル

#評価基準：対数尤度

#最適化：確率的勾配法
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support, confusion_matrix

from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler,RobustScaler

from sklearn.model_selection import train_test_split

np.random.seed(1234)
#データセットの読み込み

df_ks = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

display(df_ks.head(10))

display(df_ks.describe())

display(df_ks.isnull().sum())#欠損値

display(df_ks.dtypes)
#欠損値削除

df_ks = df_ks.dropna()

display(df_ks.isnull().sum())
#日付をdatetime型に変換し、打ち上げ'launched'から締め切り'deadline'までの日数'period'を算出

df_ks['deadline'] = pd.to_datetime(df_ks['deadline'], errors = 'coerce')

df_ks['launched'] = pd.to_datetime(df_ks['launched'], errors = 'coerce')

df_ks['period'] = (df_ks['deadline'] - df_ks['launched']).dt.days

display(df_ks.head(10))
#stateがsuccessfulとfailedのデータのみ抽出

df_ks_S = df_ks[df_ks['state']=='successful']

df_ks_F = df_ks[df_ks['state']=='failed']

df_ks2 = pd.concat([df_ks_S,df_ks_F])
#カテゴリ変数を数値データに変換

le =LabelEncoder()

le = le.fit(df_ks2['state'])

df_ks2['state'] = le.transform(df_ks2['state'])

display(df_ks2.head(10))
#objectデータを数値データに変換

df_ks2['goal'] = pd.to_numeric(df_ks2['goal'], errors ='coerce')

df_ks2['pledged'] = pd.to_numeric(df_ks2['pledged'], errors ='coerce')

df_ks2['backers'] = pd.to_numeric(df_ks2['backers'], errors ='coerce')

df_ks2['usd pledged'] = pd.to_numeric(df_ks2['usd pledged'], errors ='coerce')
#不要変数を削除

df_ks3 = df_ks2.drop(['ID','name','deadline','launched','currency','backers','pledged','usd pledged'], axis=1)



#不要変数をさらに削除

df_ks4 = df_ks3.drop(['category','goal','usd_pledged_real'], axis=1)



#並び替え

df_ks5 = df_ks4.iloc[:,[2,0,1,4,3]]



display(df_ks5.head(10))
#main_categoryとcurrencyをダミー変換

df_dummy_mc = pd.get_dummies(df_ks5['main_category'])

df_dummy_cu = pd.get_dummies(df_ks5['country'])

df_ks6 = pd.concat([df_ks5.drop(['main_category'],axis=1),df_dummy_mc],axis=1)

df_ks7 = pd.concat([df_ks6.drop(['country'],axis=1),df_dummy_cu],axis=1)

display(df_ks7.head(10))
sns.heatmap(df_ks2.corr())

plt.show()
#ホールドアウト法（訓練データとテストデータに分割）

y = df_ks7['state'].values #'state' の値をyに代入する

X = df_ks7.drop('state', axis=1).values #'state'以外の変数をXに代入する

test_size = 0.3 #テストデータの割合を決める

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1234)



#標準化

std = StandardScaler()

X_train_std = std.fit_transform(X_train) #訓練データの標準化

X_test_std = std.fit_transform(X_test) #テストデータの標準化



#正規化

mm = MinMaxScaler()

X_train_mm = mm.fit_transform(X_train)

X_test_mm = mm.fit_transform(X_test) 

print(X)

print(X_train_std)

print(X_train_mm)
#学習の実行

clf = SGDClassifier(loss='log', penalty='none', max_iter=10000, fit_intercept=True,tol=1e-3)

#loss：損失関数　max_iter：学習の最大回数　fit_intercept：切片を求める

clf_std = clf.fit(X_train_std, y_train)

clf_mm = clf.fit(X_train_mm, y_train)
#結果の表示（標準化）

y_prd_test_std = clf_std.predict(X_test_std)

cm = confusion_matrix(y_test, y_prd_test_std,labels=[1,0])#1=positive(successful),0=negative(failed)

print(cm) #混同行列

print('対数尤度 = {:.3f}'.format(-log_loss(y_test,y_prd_test_std))) #対数尤度を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test_std))) #正答率を表示

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test_std) #適合率・再現率・F1値を計算

print('適合率（Precision） = {:.3f}%'.format(100 * precision[1])) #適合率を表示

print('再現率（Recall） = {:.3f}%'.format(100 * recall[1])) #再現率を表示

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1])) #F1値を表示
#結果の確認（標準化）

print(precision)#適合率

print(recall)#再現率

print(f1_score)#F1値

#[0]列：failedの結果、[1]列：successfulの結果
#結果の表示（正規化）

y_prd_test_mm = clf_mm.predict(X_test_mm)

cm= confusion_matrix(y_test, y_prd_test_mm,labels=[1,0])#1=positive(successful),0=negative(failed)

print(cm) #混同行列

print('対数尤度 = {:.3f}'.format(-log_loss(y_test,y_prd_test_mm))) #対数尤度を表示

print('正答率（Accuracy） = {:.3f}%'.format(100 * accuracy_score(y_test, y_prd_test_mm))) #正答率を表示

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_prd_test_mm) #適合率・再現率・F1値を計算

print('適合率（Precision） = {:.3f}%'.format(100 * precision[1])) #適合率を表示

print('再現率（Recall） = {:.3f}%'.format(100 * recall[1])) #再現率を表示

print('F1値（F1-score） = {:.3f}%'.format(100 * f1_score[1])) #F1値を表示
#結果の確認（正規化）

print(precision)#適合率

print(recall)#再現率

print(f1_score)#F1値

#[0]列：failedの結果、[1]列：successfulの結果