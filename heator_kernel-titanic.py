# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib as mtp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
drop_features=['PassengerId','Survived']

train_drop=train.drop(drop_features,axis=1)

train_drop.head()
train_drop.dtypes.sort_values()
train_drop.select_dtypes(include='int64').head()
train_drop.select_dtypes(include='float64').head()
train_drop.select_dtypes(include='object').head()
train.isnull().sum()[lambda x:x>0]
test.isnull().sum()[lambda x:x>0]
train.info()
train.describe()
titanic=pd.concat([train,test],sort=False)

# 记录train中的记录个数

len_train=train.shape[0]

len_train
titanic.info()
titanic['Title']=titanic.Name.apply(lambda name:name.split(',')[1].split('.')[0].strip())
titanic.head()
titanic.Title.value_counts()
List=titanic.Title.value_counts().index[4:].tolist()

mapping={}

for s in List:

    mapping[s]='Rare'

# mapping

titanic['Title']=titanic['Title'].map(lambda x:mapping[x] if x in mapping else x)
titanic.Title.value_counts()
grouped=titanic.groupby('Title')

median=grouped.Age.median()

median
def newage(cols):

    age=cols[0]

    title=cols[1]

    # 如果为空值就填充

    if pd.isnull(age):

        return median[title]

    return age



titanic.Age=titanic[['Age','Title']].apply(newage,axis=1)

titanic.info()
titanic['has_Cabin']=titanic.Cabin

titanic['has_Cabin'].loc[~titanic.Cabin.isnull()]=1

titanic['has_Cabin'].loc[titanic.Cabin.isnull()]=0

pd.crosstab(titanic.has_Cabin[:len_train],train.Survived).plot.bar(stacked=True)

# titanic.pivot_table(index=['has_Cabin'],columns=['Survived'],aggfunc=len).plot.bar(stacked=True)
titanic.Cabin=titanic.Cabin.fillna('U')

titanic[:10]
most_embarked=titanic.Embarked.value_counts().index[0]

titanic.Embarked=titanic.Embarked.fillna(most_embarked)
titanic.Fare=titanic.Fare.fillna(titanic.Fare.median())
titanic.info()
titanic.Cabin=titanic.Cabin.apply(lambda cabin:cabin[0])
titanic['Cabin'].loc[titanic.Cabin=='T']='G'
titanic.Cabin.value_counts()
pd.crosstab(titanic.Cabin[:len_train],train.Survived).plot.bar(stacked=True)
pd.crosstab(titanic.SibSp[:len_train],train.Survived).plot.bar(stacked=True)
pd.crosstab(titanic.Parch[:len_train],train.Survived).plot.bar(stacked=True)
titanic['FamilySize']=titanic.Parch+titanic.SibSp+1

pd.crosstab(titanic.FamilySize[:len_train],train.Survived).plot.bar(stacked=True)
titanic=titanic.drop(['SibSp','Parch'],axis=1)
pd.crosstab(titanic.Sex[:len_train],train.Survived).plot.bar(stacked=True)
pd.crosstab(titanic.Pclass[:len_train],train.Survived).plot.bar(stacked=True)
pd.crosstab(titanic.Title[:len_train],train.Survived).plot.bar(stacked=True)
pd.crosstab([titanic.Title[:len_train],titanic.Sex[:len_train]],train.Survived).plot.bar(stacked=True)
pd.crosstab(pd.cut(titanic.Age,8)[:len_train],train.Survived).plot.bar(stacked=True)
titanic.Age=pd.cut(titanic.Age,8,labels=False)
pd.crosstab(titanic.Embarked[:len_train],train.Survived).plot.bar(stacked=True)
titanic.head()
titanic=titanic.drop('Name',axis=1)
titanic=titanic.drop('Ticket',axis=1)
pd.crosstab(pd.qcut(titanic.Fare,4)[:len_train],train.Survived).plot.bar(stacked=True)
titanic.Fare=pd.qcut(titanic.Fare,4,labels=False)
titanic.head()
titanic.Sex=titanic.Sex.map({'male':1,'female':0})

titanic.Cabin=titanic.Cabin.map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'U':7})

titanic.Embarked=titanic.Embarked.map({'C':0,'Q':1,'S':2})

titanic.Title=titanic.Title.map({'Mr':0,'Miss':1,'Mrs':2,'Master':3,'Rare':4})
train=titanic[:len_train]

test=titanic[len_train:]
X_train=train.loc[:, 'Pclass':]

y_train=train['Survived']



X_test=test.loc[:, 'Pclass':]
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier



log_reg=LogisticRegression()

log_reg.fit(X_train, y_train)

svm_clf = SVC()

svm_clf.fit(X_train, y_train)

tree_clf=DecisionTreeClassifier()

tree_clf.fit(X_train, y_train)
print(log_reg.score(X_train,y_train))

print(svm_clf.score(X_train,y_train))

print(tree_clf.score(X_train,y_train))
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



RF=RandomForestClassifier(random_state=1)

PRF=[{'n_estimators':[10,100],'max_depth':[3,6],'criterion':['gini','entropy']}]

GSRF=GridSearchCV(estimator=RF, param_grid=PRF, scoring='accuracy',cv=2)

scores_rf=cross_val_score(GSRF,X_train,y_train,scoring='accuracy',cv=5)
model=GSRF.fit(X_train, y_train)

pred=model.predict(X_test)

print(type(pred))

# 转换类型

pred=pred.astype(np.int64)

output=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred})

print(output.info())

# output['Survived'].astype('int64')

output.to_csv("gender_submission.csv",mode='w', index=False)

print(output.info())