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
import pandas as pd

import matplotlib.pyplot as plt
# 男=0, 女=1にスイッチ

df= pd.read_csv("../input/train.csv").replace("male",0).replace("female",1)



# Ageを中央値に変換

#df["Age"].fillna(df.Age.median(), inplace=True)
ｄｆ
split_data = []

for survived in [0,1]:

    split_data.append(df[df.Survived==survived])



temp = [i["Pclass"].dropna() for i in split_data]

plt.hist(temp, histtype="barstacked", bins=3)

plt.show()
temp = [i["Embarked"].dropna() for i in split_data]

plt.hist(temp, histtype="barstacked", bins=5)

plt.show()
#temp = [i["Ticket"].dropna() for i in split_data]

#plt.hist(temp, histtype="barstacked", bins=16)

#plt.show()



#print(df)

df.loc[:,['Survived','Embarked']]
# 男:0, 女:1

df = pd.read_csv("../input/train.csv").replace("male",0).replace("female",1)

df = df.replace("male",0).replace("female",1)

df = df.replace("S",0).replace("Q",1).replace("C", 2)

# 年齢の欠損値を中央値に変換

df["Age"].fillna(df.Age.median(), inplace=True)

df["Fare"].fillna(df.Fare.median(), inplace=True)

df["Embarked"].fillna(0, inplace=True)

drop_list = ["Name", "Parch", "Cabin","Ticket", "SibSp", "Embarked"]

df2 = df.drop(drop_list, axis=1)

print(df2) 
# モデルへの入力データ生成

train_data = df2.values

X = train_data[:, 2:] # Pclass以降の変数

y  = train_data[:, 1]  # 正解データ





# グリッドサーチで分類器生成

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



parameters = {'max_depth':list(range(8,15)), 'n_estimators':[1, 10, 100], 

              "min_samples_leaf":list(range(1,3))}



# グリッドサーチ

clf = GridSearchCV(

    RandomForestClassifier(), 

    parameters, cv=5, iid=False)



# 学習

clf = clf.fit(X, y)

print(clf.score(X, y), clf.best_params_)

# 分類器の指定

clf = clf.best_estimator_
# テストデータの読み込み

test_df= pd.read_csv("../input/test.csv").replace("male",0).replace("female",1)

test_df = test_df.replace("S",0).replace("Q",1).replace("C", 2)



# 欠損値の補完

test_df["Age"].fillna(df.Age.median(), inplace=True)

test_df["Fare"].fillna(df.Fare.median(), inplace=True)

test_df["Embarked"].fillna(0, inplace=True)



test_df2 = test_df.drop(drop_list, axis=1)



test_data = test_df2.values

X_test = test_data[:, 1:]

output = clf.predict(X_test)
data_to_submit = pd.DataFrame({

    'PassengerId':test_data[:,0].astype(int),

    'Survived':output.astype(int)

})

#print(data_to_submit)

data_to_submit.to_csv('csv_to_submit.csv', index = False)
!wc -l 'csv_to_submit.csv'

!head 'csv_to_submit.csv'