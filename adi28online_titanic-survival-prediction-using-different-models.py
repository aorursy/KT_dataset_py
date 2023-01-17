# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

print(train.head())
print(train.describe())
print(pd.isnull(train).sum())

comb=[train,test]
sns.barplot(x="Sex",y="Survived", data=train)



print(train[['Survived','Sex']].groupby(['Sex']).mean())

#pd.crosstab(train['Sex'],train['Survived'])
sns.barplot(x="Pclass",y="Survived",data=train)

print(train["Survived"][train["Pclass"]==1].value_counts())

print(train["Survived"][train["Pclass"]==2].value_counts())

print(train["Survived"][train["Pclass"]==3].value_counts())

# Made a feature that if a person is alone or with family. 

# So combined parent children and Sibling spouse features.

# After combining their values if it is 0 then the person is travelling alone.



train['FamilySize']=train['Parch']+train['SibSp']

test['FamilySize']=test['Parch']+test['SibSp']



train['Alone']=0

test['Alone']=0

train.loc[train['FamilySize']==0,'Alone']=1

test.loc[test['FamilySize']==0,'Alone']=1

sns.barplot(x='Alone',y='Survived',data=train)
# In Age we have seen that there are huge number of values that are missing so we need to find a way to fill those

# As age can be one of the important features



#So first we categorise the KNOWN age into few bins with the missing(filled with -0.5) ones into the category 'UNKNOWN'



train['Age']=train['Age'].fillna(-0.5)

test['Age'] = test['Age'].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#train['AgeGroup'].unique()

#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)
print(train[['Embarked','Survived']].groupby(['Embarked']).sum())

print(train[['Embarked','Survived']].groupby(['Embarked']).mean())

sns.barplot(x='Embarked',y='Survived',data=train)
# Since a very high percentage of passengers have 'Embarked' as S we 

# are filling the missing 2 as 'S' and also mapping them



train['Embarked']=train['Embarked'].fillna('S')

train['Embarked']=train['Embarked'].map({"S":1, "C":2, "Q":3})

test['Embarked']=test['Embarked'].map({"S":1, "C":2, "Q":3})
train['Sex']=train['Sex'].map({"male":0, "female":1})

test['Sex']=test['Sex'].map({"male":0, "female":1})
train['FareBand']=pd.qcut(train['Fare'], 4, labels=[1,2,3,4])

test['FareBand']=pd.qcut(test['Fare'], 4, labels=[1,2,3,4])

sns.barplot(x='FareBand', y='Survived', data=train)
train['title']=train['Name'].str.extract(pat = '([A-Za-z]+)\.') 

test['title']=test['Name'].str.extract(pat = '([A-Za-z]+)\.') 

print(train['title'].value_counts())
#Trying to reduce the categories into more logical terms



for data in comb:

    data['title']=data['title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')

    data['title']=data['title'].replace(['Mme'],'Mrs')

    data['title']=data['title'].replace(['Mlle','Ms'],'Miss')

    data['title']=data['title'].replace(['Countess','Sir','Lady'],'Royal')



train['title'].value_counts()

pd.crosstab(train['title'],train['Survived'])
train1=train.drop(columns=['Cabin','Ticket','Age','FamilySize','SibSp','Parch','Name','Fare','PassengerId','Survived','AgeGroup','title'])

test1=test.drop(columns=['Cabin','Ticket','Age','FamilySize','SibSp','Parch','Name','Fare','PassengerId','AgeGroup','title'])

train1.head()

test1.head()
# Imputation to handle missing values

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

test1_imputed = my_imputer.fit_transform(test1)
from sklearn.model_selection import train_test_split



target = train['Survived']

x_train,x_val,y_train,y_val = train_test_split(train1,target,test_size = 0.2, random_state=0)
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(x_train,y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100,2)

print(acc_gaussian)

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val)*100, 2)

print(acc_logreg)
from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk]})

models.sort_values(by='Score', ascending=False)
sns.barplot(x='Score', y='Model', data=models.sort_values(by=["Score"]), color="y")
ids = test['PassengerId']

predictions = decisiontree.predict(test1_imputed)



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submissions.csv', index=False)