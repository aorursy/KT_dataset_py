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
test = pd.read_csv("/kaggle/input/titanic/test.csv")
print(train.shape)
print(test.shape)
train.head()
train.info()
train.describe()
#オブジェクト型の要素数、ユニーク数、最頻値の出現回数を表示
train.describe(include='O')
# import pandas_profiling as pdp
# pdp.ProfileReport(train)
sns.pairplot(train)
plt.show
display(pd.crosstab(train["Sex"],train["Survived"]))
display(pd.crosstab(train["Sex"],train["Survived"],normalize="index"))
#性別別の生存数
g = sns.countplot(x="Sex", hue="Survived", data=train)

#性別別の生存率 
g = sns.factorplot(x="Sex", y="Survived", data=train, kind="bar")


# 等級別の生存率
g = sns.factorplot(x="Pclass", y="Survived", data=train, size=6,kind="bar")
# 等級、性別別の生存率
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", size=6, data=train, kind="bar")
plt.figure(figsize=(8,6))
cmap = sns.color_palette("coolwarm", 200)
sns.heatmap(train.corr(), square=True, annot=True, cmap=cmap)
sns.pairplot(train, x_vars=['Pclass'], y_vars=['Fare'])
train.groupby("Pclass")[["Fare"]].mean()

g=sns.FacetGrid(train,col="Survived")
g=g.map(sns.distplot,"Age")
g.add_legend()
g=sns.FacetGrid(train,col="Sex",hue="Survived")
g=g.map(sns.distplot,"Age")
g.add_legend()

train.isnull().sum()
test.isnull().sum()
train["Fare"]=train["Fare"].fillna(train["Fare"].median())
train["Age"]=train["Age"].fillna(train["Age"].median())
train["Embarked"]=train["Embarked"].fillna("S")

test["Fare"]=test["Fare"].fillna(test["Fare"].median())
test["Age"]=test["Age"].fillna(test["Age"].median())
test["Embarked"]=test["Embarked"].fillna("S")

test.isnull().sum()

train = pd.get_dummies(train, columns=["Sex","Pclass","Embarked"])
test = pd.get_dummies(test, columns=["Sex","Pclass","Embarked"])

train.head()
train["FamilyNum"] = train["SibSp"] + train["Parch"]
train["hasFamily"] = train["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
train = train.drop(labels = ["SibSp"], axis = 1)
train = train.drop(labels = ["Parch"], axis = 1)

test["FamilyNum"] = test["SibSp"] + test["Parch"]
test["hasFamily"] = test["FamilyNum"].apply(lambda x : 1 if x >= 1 else 0)
test = test.drop(labels = ["SibSp"], axis = 1)
test = test.drop(labels = ["Parch"], axis = 1)

g = sns.factorplot(x="hasFamily", y="Survived", data=train,kind="bar")
#FamilyNumの生存率を確認
g=sns.FacetGrid(train,hue="Survived")
g=g.map(sns.distplot,"FamilyNum")
g.add_legend()

#不要カラム削除
train = train.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
test = test.drop(["PassengerId", "Name", "Ticket", "Cabin"],axis=1)
test.head()

from sklearn.model_selection import train_test_split

train_X = train.drop('Survived',axis = 1)
train_y = train.Survived
(X_train, X_test, y_train, y_test) = train_test_split(train_X, train_y , test_size = 0.3 , random_state = 0)

print("X_train:"+str(X_train.shape))
print("X_test:"+str(X_test.shape))
print("y_train:"+str(y_train.shape))
print("y_test:"+str(y_test.shape))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#モデルの構築
rfc = RandomForestClassifier(random_state=0)
#学習データにて学習
rfc.fit(X_train, y_train)

# テストデータにて予測
y_pred = rfc.predict(X_test)
print(y_pred)

#正解率
print(f'accuracy:{accuracy_score(y_test, y_pred)}')
#的中率
print(f'precision:{precision_score(y_test, y_pred)}')
#補足率
print(f'recall:{recall_score(y_test, y_pred)}')
#混同行列
cm = confusion_matrix(y_test, y_pred)
print(f'cm:{cm}')

y_proba = rfc.predict_proba(X_test)
print(y_proba)

y_pred = (rfc.predict_proba(X_test)[:, 0] < 0.5).astype(int)
print(f'score_Threshold=0.5')
print(f'precision:{precision_score(y_test, y_pred)}')
print(f'recall:{recall_score(y_test, y_pred)}')

y_pred = (rfc.predict_proba(X_test)[:, 0] < 0.6).astype(int)
print(f'score_Threshold=0.6')
print(f'precision:{precision_score(y_test, y_pred)}')
print(f'recall:{recall_score(y_test, y_pred)}')

y_pred = (rfc.predict_proba(X_test)[:, 0] < 0.7).astype(int)
print(f'score_Threshold=0.7')
print(f'precision:{precision_score(y_test, y_pred)}')
print(f'recall:{recall_score(y_test, y_pred)}')

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_proba[:,1])
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC curve (area = %.2f)'%auc)
plt.legend()
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.show()

plt.figure(figsize=(20,10))
plt.barh(
    X_train.columns[np.argsort(rfc.feature_importances_)],
    rfc.feature_importances_[np.argsort(rfc.feature_importances_)],
     label='RandomForestClassifier'
 )
plt.title('RandomForestClassifier feature importance')

# 【参考】スコア提出
# test_hon = pd.read_csv("/kaggle/input/titanic/test.csv")
# 'PassengerId'を抽出する(結果と結合するため)
# test_index = test_hon.loc[:, ['PassengerId']]
# x_test_hon = test.values
# y_test_hon = rfc.predict(x_test_hon)
# PassengerId のDataFrameと結果を結合する
# df_output = pd.concat([test_index, pd.DataFrame(y_test_hon, columns=['Survived'])], axis=1)
# result.csvをカレントディレクトリに書き込む
# df_output.to_csv('result.csv', index=False)