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
path='/kaggle/input/titanic/'

train = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')

test.head()
train
import seaborn as sns

from matplotlib import pyplot as plt
f,ax=plt.subplots(1,2,figsize=(20,10))

train[train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')

ax[0].set_title('Survived= 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)

train[train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')

ax[1].set_title('Survived= 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')

sns.countplot('Survived',data=train,ax=ax[1])

ax[1].set_title('Survived')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])

ax[1].set_title('Sex:Survived vs Dead')

plt.show()
sns.countplot('Pclass', hue='Survived', data=train)

plt.title('Pclass: Sruvived vs Dead')

plt.show()
train.columns
sns.barplot(x='Survived',y='Age',data=train)
plt.bar('Age','Survived', color='red', edgecolor='white', width=1)

plt.bar('Age', 'Sex', bottom='Survived', color='green', edgecolor='white', width=1)

plt.xlabel("Survived")

plt.show()
g = sns.FacetGrid(train, hue="Survived", col="Sex", margin_titles=True,

                  palette={1:"green", 0:"red"})

g=g.map(plt.scatter, "Age", "Survived",edgecolor="w").add_legend();
train.columns
train['Age'].fillna(train['Age'].mean(),inplace=True)

test['Age'].fillna(test['Age'].mean(),inplace=True)

train['Embarked'].fillna(value='S',inplace=True)

test['Fare'].fillna(test['Fare'].mean(),inplace=True)
train.info()
train['Sex']=train['Sex'].replace(['female','male'],[0,1])

train['Embarked'] = train['Embarked'].replace(['S','Q','C'],[1,2,3])
train.head()
test['Sex'] = test['Sex'].replace(['female','male'],[0,1])

test['Embarked'] = test['Embarked'].replace(['S','Q','C'],[1,2,3])

test.head()
train['family']=train['SibSp']+train['Parch']+1

test['family']=test['SibSp']+train['Parch']+1
train_df=train.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
train_df.head()
test.columns
test_df=test.drop(columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin'])
test_df.head()
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
x = train_df.drop('Survived', axis=1)

y = train_df['Survived']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=100)
svc=SVC()
svc.fit(x_train,y_train)
svc.score(x_test,y_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
rfc = RandomForestClassifier()



parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }





grid_obj = GridSearchCV(rfc, parameters, cv=5)

grid_obj = grid_obj.fit(x_train, y_train)



rfc = grid_obj.best_estimator_

 

rfc.fit(x_train, y_train)
rfc.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score
model = LogisticRegression() 

model.fit(x_train, y_train)
model.score(x_test,y_test)
from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_train,y_train)
xgb.score(x_test,y_test)