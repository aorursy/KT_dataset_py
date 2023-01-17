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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
test.head()
train.columns
train.dtypes
train.describe(include='all')
pd.isnull(train).sum()
test.columns
test.dtypes
test.describe(include='all')
pd.isnull(test).sum()
sns.countplot('Survived',data=train)

plt.show()
#train[['Pclass','Survived']].plot(kind='bar')

sns.barplot(x='Pclass', y='Survived',data=train)

plt.show()
sns.barplot(x='Sex',y='Survived',data=train)

plt.show()
bins=[0,5,12,18,24,35,60,np.inf]

labels=['Baby','Child','Teenager','Stdent','Young Adult','Adult','Senior']

train['Agegroup']=pd.cut(train['Age'],bins=bins,labels=labels)

test['Agegroup']=pd.cut(test['Age'],bins=bins,labels=labels,)

train.head()
sns.barplot(x='Agegroup',y='Survived',data=train)

plt.show()
train=train.drop(['Agegroup'],axis=1)

test=test.drop(['Agegroup'],axis=1)

sns.barplot(x='SibSp',y='Survived',data=train)

plt.show()
sns.barplot(x='Parch',y='Survived',data=train)

plt.show()
fares_survived=train['Fare'][train['Survived']==1]

fares_notsurvived=train['Fare'][train['Survived']==0]

res=pd.DataFrame([fares_notsurvived.mean(),fares_survived.mean()])

res.plot(kind='bar')

plt.show()
train['hascabin']=train['Cabin'].notnull().astype('int')

test['hascabin']=test['Cabin'].notnull().astype('int')

train.head()
sns.barplot(x='hascabin',y='Survived',data=train)

plt.plot()
sns.barplot(x='Embarked',y='Survived',data=train)

plt.plot()
train=train.drop('PassengerId',axis=1)

train.head()
test_ids=test['PassengerId']

test=test.drop('PassengerId',axis=1)

test.head()
combine=[train,test]

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'],train['Sex'])
#replace various titles with more common names

for dataset in combine:

    dataset['Title']=dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'],'Rare')

    dataset['Title']=dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace(['Ms','Mlle'], 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title','Survived']].groupby('Title',as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    #dataset=dataset.drop('Name',axis=1)

    dataset['Title']=dataset['Title'].map(title_mapping)

    dataset['Title']=dataset['Title'].fillna(0)

train=train.drop(['Name'],axis=1)

train.head()
test=test.drop(['Name'],axis=1)

test.head()
sex_mapping={'male':1,'female':0}

train['Sex']=train['Sex'].map(sex_mapping)

test['Sex']=test['Sex'].map(sex_mapping)

train.head()
train_corr = train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

train_corr[train_corr['Feature 1']=='Age']
#take the median value for Age feature based on 'Pclass' and 'Title'

train['Age'] = train.groupby(['Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

test['Age'] = test.groupby(['Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))

bins=[0,5,12,18,24,35,60,np.inf]

labels=['Baby','Child','Teenager','Student','Young Adult','Adult','Senior']

train['Agegroup']=pd.cut(train['Age'],bins=bins,labels=labels)

test['Agegroup']=pd.cut(test['Age'],bins=bins,labels=labels,)

train=train.drop(['Age'],axis=1)

test=test.drop(['Age'],axis=1)

Agegroup_mapping = {"Baby": 1, "Child": 2, "Teenager": 3,"Student":4,"Young Adult":5,"Adult":6,"Senior":7}

train['Agegroup'] = train['Agegroup'].map(Agegroup_mapping).astype('int')

test['Agegroup'] = test['Agegroup'].map(Agegroup_mapping).astype('int')

train.dtypes
train=train.drop(['Ticket'],axis=1)

test=test.drop(['Ticket'],axis=1)

train.head()
combine=[train,test]

for dataset in combine:

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

for dataset in combine:

   # print(dataset.head())

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

train.head()
train=train.drop(['Cabin'],axis=1)

test=test.drop(['Cabin'],axis=1)
#replacing the missing values in the Embarked feature with S

train = train.fillna({"Embarked": "S"})

embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
print(pd.isnull(train).sum(),'\n',train.dtypes)

#train,valid=train_test_split()
from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.svm import SVC,LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier





X = train.drop(['Survived'], axis=1)

Y = train["Survived"]



kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Support Vector Machines', 'K-Nearst Neighbor', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'XGBoost']

models=[SVC(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), GaussianNB(), 

        Perceptron(), LinearSVC(), DecisionTreeClassifier(), SGDClassifier(), 

        GradientBoostingClassifier(), XGBClassifier()]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,Y, cv = kfold,scoring = "accuracy")

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)  

new_models_dataframe2.sort_values('CV Mean',ascending=False,inplace=True)   

new_models_dataframe2
print(pd.isnull(test).sum(),'\n',test.dtypes)
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.metrics import make_scorer,accuracy_score

params = {

        'n_estimators': [60, 100, 200],

        'reg_alpha': [0,0.5, 1, 1.5, 2],

        'max_depth': [3, 4, 5, 6]

        }

xgb = XGBClassifier(learning_rate=0.05,objective='binary:logistic')

print(xgb)

grid = GridSearchCV(xgb, 

                    param_grid = params, 

                    scoring = make_scorer(accuracy_score), 

                    n_jobs = -1, 

                    cv = 5,)

                    #refit = "accuracy_score")



clf=grid.fit(X,Y)

print('score=',clf.score(X,Y))



predictions = clf.predict(test)

output = pd.DataFrame({ 'PassengerId' : test_ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)

print("The submission was successfully saved!")


