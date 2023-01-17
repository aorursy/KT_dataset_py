# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

data = pd.read_csv("../input/titanic-extended/train.csv")

df = pd.read_csv('../input/titanic-extended/test.csv')
data.head()
df.head()
df.shape, data.shape
data.isnull().sum(), df.isnull().sum()
data.info()
df.info()
data.Age.loc[data.Age.isna()] = data.Age_wiki
df.Age.loc[df.Age.isna()] = df.Age_wiki
data.Age.loc[data.Age.isna()] = data.Age.mean()
data.Embarked.loc[data.Embarked.isna()] = 'S'
df.Fare.loc[df.Fare.isna()] = 13.67
data = data.iloc[:,:12]

data.head()
data.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)

data.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data.Survived = data.Survived.astype('int64')
sns.pairplot(data[['Survived','Age','SibSp','Parch','Fare']])
data = pd.get_dummies(data,columns=['Sex','Embarked'],drop_first=True)

data.head()
X = data.drop('Survived',axis=1)

y= data.Survived
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
X_train,X_valid,y_train,y_valid = train_test_split(X,y,random_state=0,stratify=y)
lr = LogisticRegression().fit(X_train,y_train)

lr.score(X_train,y_train), lr.score(X_valid,y_valid)
from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_valid,lr.predict(X_valid))
confusion_matrix(y_train,lr.predict(X_train))
data.corr()
plt.figure(figsize=(12,12))

sns.heatmap(data.corr(),annot=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant
dff = X.assign(const=1)

vif = pd.DataFrame([variance_inflation_factor(dff.values,i) for i in range(dff.shape[1])],index=dff.columns)

vif
vif.reset_index(inplace=True)

vif.columns = ['col','val']

vif
vif.sort_values(by='val')
data.corr()['Survived']
X1 = data[['Pclass','Fare','Sex_male','Embarked_S']]

X_train1,X_valid1,y_train1,y_valid1 = train_test_split(X1,y,random_state=0,stratify=y)

lr1 = LogisticRegression().fit(X_train1,y_train1)

confusion_matrix(y_valid1,lr1.predict(X_valid1))
lr1.score(X_train1,y_train1), lr1.score(X_valid1,y_valid1)
from sklearn.ensemble import RandomForestClassifier

rr = RandomForestClassifier(n_estimators=70).fit(X_train,y_train)

confusion_matrix(y_valid,rr.predict(X_valid))
rr1 = RandomForestClassifier(n_estimators=70).fit(X_train1,y_train1)

confusion_matrix(y_valid1,rr1.predict(X_valid1))
from sklearn.naive_bayes import GaussianNB

gg = GaussianNB().fit(X_train,y_train)

confusion_matrix(y_valid,gg.predict(X_valid))
gg1 = GaussianNB().fit(X_train1,y_train1)

confusion_matrix(y_valid1,gg1.predict(X_valid1))
from sklearn.metrics import f1_score

sc = []

li = [x for x in range(150,250,10)]

for i in li:

    rr2 = RandomForestClassifier(n_estimators=i,random_state=0).fit(X_train,y_train)

    sc.append(f1_score(y_valid,rr2.predict(X_valid)))

sns.regplot(li,sc)
rr = RandomForestClassifier(n_estimators=150,random_state=0).fit(X_train,y_train)

confusion_matrix(y_valid,rr.predict(X_valid))
from xgboost import XGBClassifier

xx = XGBClassifier().fit(X_train,y_train)

confusion_matrix(y_valid,xx.predict(X_valid))
from sklearn.svm import SVC

ss = SVC().fit(X_train,y_train)

confusion_matrix(y_valid,ss.predict(X_valid))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train,y_train)

confusion_matrix(y_valid,knn.predict(X_valid))
df = df.iloc[:,:12]

df.head()
df.drop(['Name','Ticket','Cabin','WikiId'],axis=1,inplace =True)

df = pd.get_dummies(df,columns=['Sex','Embarked'],drop_first=True)

df.head()
pred = rr.predict(df)

pred[:5]
Submission = pd.DataFrame({'PassengerId': df['PassengerId'], 'Survived': pred})

Submission.to_csv('Submission.csv',index=False)