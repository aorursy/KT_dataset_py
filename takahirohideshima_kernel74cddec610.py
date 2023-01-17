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
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import codecs as cd

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import log_loss, confusion_matrix

import seaborn as sns

from sklearn.model_selection import train_test_split # ホールドアウト法に関する関数

from sklearn.model_selection import KFold # 交差検証法に関する関数

from sklearn.metrics import mean_absolute_error # 回帰問題における性能評価に関する関数
#データセット確認

data = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", header = 0)

print(data.shape)

data.head()
#予測に使えないデータの削除　

#カテゴリーはメインカテゴリーがあるため削除　ゴールはusdゴールがあるため削除

drop_elements = ['usd_pledged_real','usd pledged','pledged','backers','ID','name','goal','category']

data2 = data.drop(drop_elements, axis = 1)

data2.head()
#統計値を表示

display(data2.describe())

#欠損値がある行数

display(data2.isnull().sum())

#データタイプ

display(data2.dtypes)
#DAY2で追加

#usd_goal_realの調整

#histogram

sns.distplot(data2['usd_goal_real']);
#usd_goal_realの調整   対数をとる

data2['usd_goal_real'] = np.log(data2['usd_goal_real'] )

sns.distplot(data2['usd_goal_real']);



#stateの要素の個数

data2['state'].value_counts()
#日時に変換,期間を追加

data2['deadline'] = pd.to_datetime(data2['deadline'], errors = 'coerce')

data2['launched'] = pd.to_datetime(data2['launched'], errors = 'coerce')

data2['period'] = (data2['deadline'] - data2['launched']).dt.days

data2 = data2.drop(['deadline', 'launched'], axis=1)

#histogram

sns.distplot(data2['period']);

display(data2.describe())
#DAY2で追加

#periodの外れ値を削除

data2 = data2[ (data2['period'] < 1000)]

data2 = data2[ (data2['period'] > 1)]

#usd_goal_real	の外れ値を削除

data2 = data2[ (data2['usd_goal_real'] > 1)]

#histogram

sns.distplot(data2['period']);

display(data2.describe())



#並び替え

data2 = data2.iloc[:,[2,4,5,0,1,3]]

data2.head()
#DAY3で追加

#新たな説明変数

data2['Goal/Period'] = np.log(data2['usd_goal_real']/data2['period'])

data2.head()

display(data2.describe())

sns.distplot(data2['Goal/Period']);
#結果の確認

data2['state'].unique()
#成功と失敗のみ抽出

data2 = data2[(data2['state'] == 'successful') | (data2['state'] == 'failed')]

data2['state'].unique()
#ダミー

cate_dummy = pd.get_dummies(data2['main_category'])

data2 = pd.concat([data2.drop(['main_category'],axis=1),cate_dummy],axis=1)

curr_dummy = pd.get_dummies(data2['currency'])

data2 = pd.concat([data2.drop(['currency'],axis=1),curr_dummy],axis=1)

#国は通貨があるため削除

data2 = data2.drop(['country'], axis=1)

print(data2.shape)
#目的変数と説明変数

y = data2['state'].map({'successful': 1, 'failed': 0})

X = data2.drop('state', axis=1)

#DAY2で追加 ホールドアウト法  test30%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

display(X_train.describe())

display(X_test.describe())

display(y_train.describe())

display(y_test.describe())
#DAY2で追加

#usd_goal_realとperiodを正規化

# 正規化

X_train["N_usd_goal_real"] = (X_train["usd_goal_real"] - X_train["usd_goal_real"].min()) / (X_train["usd_goal_real"].max() - X_train["usd_goal_real"].min())

X_test["N_usd_goal_real"] = (X_test["usd_goal_real"] - X_train["usd_goal_real"].min()) / (X_train["usd_goal_real"].max() - X_train["usd_goal_real"].min())

X_train["N_period"] = (X_train["period"] - X_train["period"].min()) / (X_train["period"].max() - X_train["period"].min())

X_test["N_period"] = (X_test["period"] - X_train["period"].min()) / (X_train["period"].max() - X_train["period"].min())

X_train = X_train.drop('period', axis=1)

X_test = X_test.drop('period', axis=1)

X_train = X_train.drop('usd_goal_real', axis=1)

X_test = X_test.drop('usd_goal_real', axis=1)



display(X_train.describe())

display(X_test.describe())

#ロジスティック回帰で学習

from sklearn.linear_model import LinearRegression, LogisticRegression

lr = LogisticRegression() 

lr.fit(X_train, y_train)
# 学習した結果を使って説明変数を入力して予測

y_est = lr.predict(X_test)



import numpy as np

import matplotlib.pyplot as plt

 

# ヒストグラムを出力

plt.hist(y_est)



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

print("accuracy_score :{:.2}  ".format( accuracy_score(y_test, y_est)))

print("precision_score :{:.2}  ".format( precision_score(y_test, y_est)))

print("recall_score :{:.2}  ".format( recall_score(y_test, y_est)))

print("f1_score:{:.2}  ".format( f1_score(y_test, y_est) ))



#DAY3で追加

#トレーニング誤差の確認

y_estrain = lr.predict(X_train)

print("Train_accuracy_score :{:.2}  ".format( accuracy_score(y_train, y_estrain)))

print("Train_precision_score :{:.2}  ".format( precision_score(y_train, y_estrain)))

print("Train_recall_score :{:.2}  ".format( recall_score(y_train, y_estrain)))

print("Train_f1_score:{:.2}  ".format( f1_score(y_train, y_estrain)))

 

#DAY2で追加

#正則化

from sklearn.linear_model import LinearRegression, LogisticRegression

lr2 = LogisticRegression(C=0.001, penalty='l1')

lr2.fit(X_train, y_train)


y_est2 = lr2.predict(X_test)

print("accuracy_score :{:.2}  ".format( accuracy_score(y_test, y_est2)))

print("precision_score :{:.2}  ".format( precision_score(y_test, y_est2)))

print("recall_score :{:.2}  ".format( recall_score(y_test, y_est2)))

print("f1_score:{:.2}  ".format( f1_score(y_test, y_est2) ))



#DAY3で追加

#トレーニング誤差の確認

y_estrain2 = lr2.predict(X_train)

print("Train_accuracy_score :{:.2}  ".format( accuracy_score(y_train, y_estrain2)))

print("Train_precision_score :{:.2}  ".format( precision_score(y_train, y_estrain2)))

print("Train_recall_score :{:.2}  ".format( recall_score(y_train, y_estrain2)))

print("Train_f1_score:{:.2}  ".format( f1_score(y_train, y_estrain2)))

 
#DAY2で追加

#正則化

from sklearn.linear_model import LinearRegression, LogisticRegression

lr3 = LogisticRegression(C=0.001, penalty='l2')

lr3.fit(X_train, y_train)
y_est3 = lr3.predict(X_test)

print("accuracy_score :{:.2}  ".format( accuracy_score(y_test, y_est3)))

print("precision_score :{:.2}  ".format( precision_score(y_test, y_est3)))

print("recall_score :{:.2}  ".format( recall_score(y_test, y_est3)))

print("f1_score:{:.2}  ".format( f1_score(y_test, y_est3) ))



#DAY3で追加

#トレーニング誤差の確認

y_estrain3 = lr3.predict(X_train)

print("Train_accuracy_score :{:.2}  ".format( accuracy_score(y_train, y_estrain3)))

print("Train_precision_score :{:.2}  ".format( precision_score(y_train, y_estrain3)))

print("Train_recall_score :{:.2}  ".format( recall_score(y_train, y_estrain3)))

print("Train_f1_score:{:.2}  ".format( f1_score(y_train, y_estrain3)))
#DAY3で追加

#random forest

from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=30, max_depth=50, criterion="gini",

                                                 min_samples_leaf=3, min_samples_split=2, random_state=1234)

clf.fit(X_train, y_train)

#print("score=", clf.score(X_train, y_train))



y_est4 = clf.predict(X_test)

print("accuracy_score :{:.2}  ".format( accuracy_score(y_test, y_est4)))

print("precision_score :{:.2}  ".format( precision_score(y_test, y_est4)))

print("recall_score :{:.2}  ".format( recall_score(y_test, y_est4)))

print("f1_score:{:.2}  ".format( f1_score(y_test, y_est4) ))



#トレーニング誤差の確認

y_estrain4 =  clf.predict(X_train)

print("Train_accuracy_score :{:.2}  ".format( accuracy_score(y_train, y_estrain4)))

print("Train_precision_score :{:.2}  ".format( precision_score(y_train, y_estrain4)))

print("Train_recall_score :{:.2}  ".format( recall_score(y_train, y_estrain4)))

print("Train_f1_score:{:.2}  ".format( f1_score(y_train, y_estrain4)))