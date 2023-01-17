# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv("/kaggle/input/titanic/train.csv")

#TESTのデータは今回使用しない。
#test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.isnull().sum()
train.head(6)
print(train[train['Age'].isnull()])
train.describe()
train.hist("Age")
plt.tight_layout()
plt.show()
train.boxplot(column="Age")
train1 = train.dropna(subset=['Age'])
train1.head(6)
train2= train.fillna({'Age': 0 })
train2.head(6)
train3=train.fillna(train.mean())
train3.head(6)
train4=train.fillna(train.median())
train4.head(6)
train = pd.read_csv("/kaggle/input/titanic/train.csv")
train5 = train

#平均・標準偏差・null数を取得する
Age_average = 29.69
Age_std = 14.52
Age_nullcount = 177

# 正規分布に従うとし、標準偏差の範囲内でランダムに数字を作る
rand = np.random.randint(Age_average - Age_std, Age_average + Age_std , size = Age_nullcount)

#Ageの欠損値をランダムな値で埋める
train5["Age"][np.isnan(train5["Age"])] = rand

train5.head(6)

train6 = pd.read_csv("/kaggle/input/titanic/train.csv")

#trainデータ
train6.loc[(train6['Name'].str.contains('Mr\.')) & (train6['Age'].isnull()), 'Age'] = train[train['Name'].str.contains('Mr\.')].Age.mean()
train6.loc[(train6['Name'].str.contains('Mrs\.')) & (train6['Age'].isnull()), 'Age'] = train[train['Name'].str.contains('Mrs\.')].Age.mean()
train6.loc[(train6['Name'].str.contains('Miss\.')) & (train6['Age'].isnull()), 'Age'] = train[train['Name'].str.contains('Miss\.')].Age.mean()
train6.loc[(train6['Name'].str.contains('Master\.')) & (train6['Age'].isnull()), 'Age'] = train[train['Name'].str.contains('Master\.')].Age.mean()
train6.loc[(train6['Name'].str.contains('Dr\.')) & (train6['Age'].isnull()), 'Age'] = train[train['Name'].str.contains('Dr\.')].Age.mean()

train6.head(6)
train["Fare"]=train["Fare"].fillna(train["Fare"].median())
train["Embarked"]=train["Embarked"].fillna("S")


# #testデータのAgeは中央値で欠損処理
# test["Fare"]=test["Fare"].fillna(test["Fare"].median())
# test["Age"]=test["Age"].fillna(test["Age"].median())
# test["Embarked"]=test["Embarked"].fillna("S")

# test.isnull().sum()

train = pd.get_dummies(train, columns=["Sex","Pclass","Embarked"])
train1 = pd.get_dummies(train2, columns=["Sex","Pclass","Embarked"])
train2 = pd.get_dummies(train2, columns=["Sex","Pclass","Embarked"])
train3 = pd.get_dummies(train3, columns=["Sex","Pclass","Embarked"])
train4 = pd.get_dummies(train4, columns=["Sex","Pclass","Embarked"])
train5 = pd.get_dummies(train5, columns=["Sex","Pclass","Embarked"])
train6 = pd.get_dummies(train6, columns=["Sex","Pclass","Embarked"])

# test = pd.get_dummies(test, columns=["Sex","Pclass","Embarked"])

train["FamilyNum"] = train["SibSp"] + train["Parch"]
train1["FamilyNum"] = train1["SibSp"] + train1["Parch"]
train2["FamilyNum"] = train2["SibSp"] + train2["Parch"]
train3["FamilyNum"] = train3["SibSp"] + train3["Parch"]
train4["FamilyNum"] = train4["SibSp"] + train4["Parch"]
train5["FamilyNum"] = train5["SibSp"] + train5["Parch"]
train6["FamilyNum"] = train6["SibSp"] + train6["Parch"]

train["hasFamily"] = train["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train1["hasFamily"] = train1["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train2["hasFamily"] = train2["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train3["hasFamily"] = train3["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train4["hasFamily"] = train4["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train5["hasFamily"] = train5["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train6["hasFamily"] = train6["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)

train = train.drop(labels = ["SibSp"], axis = 1)
train1 = train1.drop(labels = ["SibSp"], axis = 1)
train2 = train2.drop(labels = ["SibSp"], axis = 1)
train3 = train3.drop(labels = ["SibSp"], axis = 1)
train4 = train4.drop(labels = ["SibSp"], axis = 1)
train5 = train5.drop(labels = ["SibSp"], axis = 1)
train6 = train6.drop(labels = ["SibSp"], axis = 1)

train = train.drop(labels = ["Parch"], axis = 1)
train1 = train1.drop(labels = ["Parch"], axis = 1)
train2 = train2.drop(labels = ["Parch"], axis = 1)
train3 = train3.drop(labels = ["Parch"], axis = 1)
train4 = train4.drop(labels = ["Parch"], axis = 1)
train5 = train5.drop(labels = ["Parch"], axis = 1)
train6 = train6.drop(labels = ["Parch"], axis = 1)

# test["FamilyNum"] = test["SibSp"] + test["Parch"]
# test["hasFamily"] = test["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
# test = test.drop(labels = ["SibSp"], axis = 1)
# test = test.drop(labels = ["Parch"], axis = 1)
 
#不要カラム削除
train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
train1 = train1.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
train2 = train2.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
train3 = train3.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
train4 = train4.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
train5 = train5.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
train6 = train6.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)

# test = test.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)

train1.describe()

from sklearn.model_selection import train_test_split

train_X = train.drop('Survived',axis = 1)
train_X1 = train1.drop('Survived',axis = 1)
train_X2 = train2.drop('Survived',axis = 1)
train_X3 = train3.drop('Survived',axis = 1)
train_X4 = train4.drop('Survived',axis = 1)
train_X5 = train5.drop('Survived',axis = 1)
train_X6 = train6.drop('Survived',axis = 1)

train_y = train.Survived
train_y1 = train1.Survived
train_y2 = train2.Survived
train_y3 = train3.Survived
train_y4 = train4.Survived
train_y5 = train5.Survived
train_y6 = train6.Survived


(X_train, X_test, y_train, y_test) = train_test_split(train_X, train_y , test_size = 0.3 , random_state = 0)
(X_train1, X_test1, y_train1, y_test1) = train_test_split(train_X1, train_y1 , test_size = 0.3 , random_state = 0)
(X_train2, X_test2, y_train2, y_test2) = train_test_split(train_X2, train_y2 , test_size = 0.3 , random_state = 0)
(X_train3, X_test3, y_train3, y_test3) = train_test_split(train_X3, train_y3 , test_size = 0.3 , random_state = 0)
(X_train4, X_test4, y_train4, y_test4) = train_test_split(train_X4, train_y4 , test_size = 0.3 , random_state = 0)
(X_train5, X_test5, y_train5, y_test5) = train_test_split(train_X5, train_y5 , test_size = 0.3 , random_state = 0)
(X_train6, X_test6, y_train6, y_test6) = train_test_split(train_X6, train_y6 , test_size = 0.3 , random_state = 0)


print("X_train:"+str(X_train1.shape))
# print("X_test:"+str(X_test1.shape))
print("y_train:"+str(y_train1.shape))
# print("y_test:"+str(y_test1.shape))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#モデルの構築
rfc1 = RandomForestClassifier(random_state=0)
rfc2 = RandomForestClassifier(random_state=0)
rfc3 = RandomForestClassifier(random_state=0)
rfc4 = RandomForestClassifier(random_state=0)
rfc5 = RandomForestClassifier(random_state=0)
rfc6 = RandomForestClassifier(random_state=0)

#学習データにて学習
rfc1.fit(X_train1, y_train1)
rfc2.fit(X_train2, y_train2)
rfc3.fit(X_train3, y_train3)
rfc4.fit(X_train4, y_train4)
rfc5.fit(X_train5, y_train5)
rfc6.fit(X_train6, y_train6)

# テストデータにて予測
y_pred1 = rfc1.predict(X_test1)
y_pred2 = rfc2.predict(X_test2)
y_pred3 = rfc3.predict(X_test3)
y_pred4 = rfc4.predict(X_test4)
y_pred5 = rfc5.predict(X_test5)
y_pred6 = rfc6.predict(X_test6)


#正解率
print("正解率")
print(f'accuracy1:{accuracy_score(y_test1, y_pred1)}')
print(f'accuracy2:{accuracy_score(y_test2, y_pred2)}')
print(f'accuracy3:{accuracy_score(y_test3, y_pred3)}')
print(f'accuracy4:{accuracy_score(y_test4, y_pred4)}')
print(f'accuracy5:{accuracy_score(y_test5, y_pred5)}')
print(f'accuracy6:{accuracy_score(y_test6, y_pred6)}')

