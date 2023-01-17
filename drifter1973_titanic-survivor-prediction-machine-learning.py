# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head(10)
train_data.describe()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head(10)
test_data.describe()
train_data1 = train_data.drop(columns=['PassengerId','Ticket','Cabin'])
train_data1.head()
test_data1 = test_data.drop(columns=['Ticket','Cabin'])
test_data1.head()
combine = [train_data1,test_data1]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False) #Regular expression
    
pd.crosstab(train_data1['Title'], train_data1['Sex']) #crosstab:caculate the frequency of each groups
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Rev','Sir'],'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss') #Mlle is Miss in French
    dataset['Title'] = dataset['Title'].replace('Ms','Miss') #Ms is abbreviation of Miss
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs') #Mme is Mrs in French
    
train_data1[['Title', 'Survived']].groupby(['Title'] ,as_index=False).mean()
title_mapping = {"Mr":1,"Mrs":2,"Miss":3,"Master":4,"Rare":5}

for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
train_data1.head()
train_data1 = train_data1.drop(columns=['Name'])
test_data1 = test_data1.drop(columns=['Name'])
combine=[train_data1,test_data1]
train_data1.shape, test_data1.shape
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {"male":1,"female":0} ).astype(int)
    
train_data1.head()
# grid = sns.FacetGrid(train_data1,col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_data1, row='Pclass', col='Sex', size=4, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
lack = train_data1.isnull()
lack_col = lack.any()
lack_col
lack1 = test_data1.isnull()
lack_col1 = lack1.any()
lack_col1
train_data1['Age']=train_data1.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median())).astype(int) #use median of age to fill with groupby selection
train_data1
test_data1['Age']=test_data1.groupby(['Sex','Pclass'])['Age'].apply(lambda x:x.fillna(x.median())).astype(int) #use median of age to fill with groupby selection
test_data1
train_data1.groupby('Embarked')['Survived'].count()
train_data1['Embarked'] = train_data1['Embarked'].fillna('S')
train_data1
combine = [train_data1,test_data1]
# grouping age
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 64), 'Age'] = 4
train_data1.head()
train_data1[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
#grouping numbers of sibsp and parch

for dataset in combine:
    dataset['Familysize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
train_data1[['Familysize','Survived']].groupby(['Familysize'],as_index = False).mean()
# make a new feature-IsAlone to distinguish single passenger from those with family

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['Familysize'] == 1, 'IsAlone'] = 1

train_data1[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_data1 = train_data1.drop(columns=['Familysize','Parch','SibSp'])
test_data1 = test_data1.drop(columns=['Familysize','Parch','SibSp'])

combine = [train_data1,test_data1]
train_data1.head()
#Embarked's mapping

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

train_data1.head()
train_data1[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
test_data1['Fare'].fillna(test_data1['Fare'].dropna().median(), inplace=True)
test_data1['Title'] = test_data1['Title'].astype(int)
test_data1.head()
#find Quartile of fare and Divide fare into four sections

from scipy .stats.mstats import mquantiles
print('Q1,Q2,Q3',mquantiles(test_data1['Fare']))
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.8958, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.8958) & (dataset['Fare'] <= 14.4542), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.4542) & (dataset['Fare'] <= 31.5), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31.5), 'Fare']=3
    dataset['Fare']=dataset['Fare'].astype(int)
    
combine = [train_data1,test_data1]

train_data1.head()
train_data1[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean()
train_data1[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
corr = train_data1.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, square=True, annot=True)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
X_train = train_data1.drop('Survived',axis=1)
y_train = train_data1['Survived']
X_test = test_data1.drop('PassengerId',axis=1)
X_train.shape, y_train.shape, X_test.shape
#Logistic Regression
logre = LogisticRegression()
logre.fit(X_train,y_train)
y_pred_logre = logre.predict(X_test)
acc_log = round(logre.score(X_train,y_train)*100,2)
print(acc_log)
#SVC
svc = SVC()
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train,y_train)*100,2)
print(acc_svc)
#KNN
knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train,y_train)*100,2)
print(acc_knn)
#Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred_gnb = gnb.predict(X_test)
acc_gnb = round(gnb.score(X_train,y_train)*100,2)
print(acc_gnb)
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
y_pred_dt = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train,y_train)*100,2)
print(acc_decision_tree)
#Random Forest
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', oob_score=True)
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
acc_rfc = round(rfc.score(X_train,y_train)*100,2)

print(acc_rfc)
print("oob_score(accuary):",rfc.oob_score_)
# Decide which model is the best
models = pd.DataFrame({
    'Model':['Logistic Regression','Support Vector Machine','KNN',
            'Random Forest','Naive Bayes',
            'Decision Tree'],
    'Score':[acc_log,acc_svc,acc_knn,
             acc_rfc,acc_gnb,
             acc_decision_tree]})
models.sort_values(by='Score',ascending=False).reset_index(drop=True)
output = pd.DataFrame({
    "PassengerId":test_data1["PassengerId"],
    "Survived":y_pred_rfc
})
output.to_csv('my_submission.csv', index=False)