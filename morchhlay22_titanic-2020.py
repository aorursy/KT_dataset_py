import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.describe(include="all")
train.sample(5)
train.isnull().sum()
sns.barplot(x="Sex",y="Survived",data=train)
print("precentage of female who survived :", train['Survived'][train['Sex']=='female'].value_counts(normalize=True)[1]*100)
print("precentage of male who survived :", train['Survived'][train['Sex']=='male'].value_counts(normalize=True)[1]*100)
print("precentage of pclas=1,who survived :", train['Survived'][train['Pclass']==1].value_counts(normalize=True)[1]*100)
print("precentage of pclas=2,who survived :", train['Survived'][train['Pclass']==2].value_counts(normalize=True)[1]*100)
print("precentage of pclas=3,who survived :", train['Survived'][train['Pclass']==3].value_counts(normalize=True)[1]*100)
sns.barplot(x="SibSp",y="Survived",data=train)
print("precentage of sibsp =0,who survived:",train['Survived'][train['SibSp']==0].value_counts(normalize=True)[1]*100)
print("precentage of sibsp = 1,who survived:",train['Survived'][train['SibSp']==1].value_counts(normalize=True)[1]*100)
print("precentage of sibsp = 2,who survived:",train['Survived'][train['SibSp']==3].value_counts(normalize=True)[1]*100)
sns.barplot(x="Parch",y="Survived",data=train)
#age feature

train['Age'] = train['Age'].fillna(-0.5)

test['Age']  = test['Age'].fillna(-0.5)

bins=[-1,0,5,12,18,24,35,60,np.inf]

labels=["unknown","baby","child","teenager","student","young_adult","adult","senior"]

train['AgeGroup']=pd.cut(train["Age"],bins,labels=labels)

test['AgeGroup']=pd.cut(test["Age"],bins,labels=labels)
plt.figure(figsize=[10,10])

sns.barplot(x="AgeGroup",y="Survived",data=train)

plt.show()
train["CabinBool"]=(train["Cabin"].notnull().astype('int'))

test['CabinBool']=(test['Cabin'].notnull().astype('int'))

                    
print("% of cabinbool = 1 who survived :" , train['Survived'][train['CabinBool']==1].value_counts(normalize=True)[1]*100)
print("% of cabinbool = 0 who survived :" , train['Survived'][train['CabinBool']==0].value_counts(normalize=True)[1]*100)
sns.boxplot(x="CabinBool",y="Survived",data=train)

plt.show()
test.describe(include="all")
train.drop("Cabin",inplace=True,axis=1)
test.drop("Cabin",inplace=True,axis=1)
train.drop("Ticket",inplace=True,axis=1)

test.drop("Ticket",inplace=True,axis=1)
print("number of people southhampton:")

southampton=train[train['Embarked']=='S'].shape[0]

print(southampton)
print("number of people cherboury:")

cherboury=train[train['Embarked']=='C'].shape[0]

print(cherboury)
print("number of people cherboury:")

queenstown=train[train['Embarked']=='Q'].shape[0]

print(queenstown)
train=train.fillna({'Embarked':"S"})
train.isnull().sum()
test.isnull().sum()
combine = [train,test]

for dataset in combine:

    dataset['Title']=dataset.Name.str.extract(pat='([A-Za-z]+)\.' ,expand=False)

pd.crosstab(train['Title'],train['Sex'])    
for dataset in combine :

    dataset['Title'] = dataset['Title'].replace(['Lady','Capt', 'Col','Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')

    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Sir'],'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle',"Miss")

    dataset['Title'] = dataset['Title'].replace('Ms','Miss')

    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

    
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()
title_mapping = {'Mr':1,'Miss':2, 'Mrs':3,'Master':4,'Royal':5,'Rare':6}

for dataset in combine:

    dataset['Title']=dataset['Title'].map(title_mapping)

    dataset['Title']=dataset['Title'].fillna(0)

train.head()    
train=train.drop(['Name'],axis=1)

test = test.drop(['Name'],axis=1)
sex_mapping = {'male':0,'female':1}

train['Sex']=train['Sex'].map(sex_mapping)

test['Sex']=test['Sex'].map(sex_mapping)

train.head()   
embarked_mapping = {"S":1,"C":2,"Q":3}

train['Embarked'] =train['Embarked'].map(embarked_mapping)

test['Embarked']  =test['Embarked'].map(embarked_mapping)
train.head()
agegroup_mapping = {"unknown":1,"baby":2,"child":3,"teenager":4,"student":5,"young_adult":6,"adult":7,"senior":8}

train['AgeGroup'] = train['AgeGroup'].map(agegroup_mapping)

test['AgeGroup']  = test['AgeGroup'].map(agegroup_mapping)
train.head()
train.drop(['Fare'],axis=1,inplace=True)

train.drop(['Age'],axis=1,inplace=True)

train.head()
feature_scale = [features for features in train.columns if features not in ["PassengerId","Survived"]]

                 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(train[feature_scale])
train = pd.concat([train[["PassengerId","Survived"]].reset_index(drop=True),

pd.DataFrame(scaler.transform(train[feature_scale]),columns=feature_scale)],axis=1)
train.head()
feature_scale = [features for features in test.columns if features not in ["PassengerId"]]
scaler = MinMaxScaler()

scaler.fit(test[feature_scale])
test = pd.concat([test[["PassengerId"]].reset_index(drop=True),

pd.DataFrame(scaler.transform(test[feature_scale]),columns=feature_scale)],axis=1)
test.head()
from sklearn.model_selection import train_test_split
X =train.drop(['Survived','PassengerId'],axis=1)

y =train['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.22,random_state=0)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

regression = LogisticRegression()

regression.fit(X_train,y_train)
y_pred = regression.predict(X_test)

acc_regression = round(accuracy_score(y_test, y_pred)*100,2)
print(acc_regression)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,y_train)

y_pred = nb.predict(X_test)

acc_nb = round(accuracy_score(y_test,y_pred)*100,2)

print(acc_nb)
from sklearn.svm import SVC

svm = SVC(kernel='rbf',C=1,gamma=0.1)

svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

acc_svm = round(accuracy_score(y_test,y_pred)*100,2)

print(acc_svm)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train,y_train)

y_pred = perceptron.predict(X_test)

acc_perceptron = round(accuracy_score(y_test,y_pred)*100,2)

print(acc_perceptron)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)

acc_DecisionTreeClassifier=round(accuracy_score(y_test,y_pred)*100,2)

print(acc_DecisionTreeClassifier)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier(n_estimators=100)

randomforest.fit(X_train,y_train)

y_pred = randomforest.predict(X_test)

acc_randomforest = round(accuracy_score(y_test,y_pred)*100,2)

print(acc_randomforest)
from sklearn.cluster import KMeans

kmeans = KMeans()

kmeans.fit(X_train,y_train)

y_pred = kmeans.predict(X_test)

acc_kmeans = round(accuracy_score(y_test,y_pred)*100,2)

print(acc_kmeans)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc_knn = round(accuracy_score(y_test,y_pred) * 100, 2)

print(acc_knn)
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

y_pred = sgd.predict(X_test)

acc_sgd = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_sgd)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(X_train, y_train)

y_pred = gbk.predict(X_test)

acc_gbk = round(accuracy_score(y_pred, y_test) * 100, 2)

print(acc_gbk)
from sklearn.model_selection import GridSearchCV

C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel=['rbf','linear']

hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd=GridSearchCV(estimator=svm,param_grid=hyper,verbose=True)

gd.fit(X_train,y_train)

print(gd.best_score_)

print(gd.best_estimator_)
n_estimators=range(100,1000,100)

hyper={'n_estimators':n_estimators}

gd=GridSearchCV(estimator=randomforest,param_grid=hyper,verbose=True)

gd.fit(X_train,y_train)

print(gd.best_score_)

print(gd.best_estimator_)
import xgboost as xg

xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)

xgboost.fit(X_train,y_train)

y_predx = xgboost.predict(X_test)

acc_xg = round(accuracy_score(y_predx, y_test) * 100, 2)

print(acc_xg)
models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron',  

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svm,acc_knn,acc_regression,acc_randomforest,acc_nb,acc_perceptron,acc_DecisionTreeClassifier,acc_sgd,acc_gbk]})

models.sort_values(by='Score', ascending=False)
test.drop(['Fare'],axis=1,inplace=True)
test.drop(['Age'],axis=1,inplace=True)
ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)
output
