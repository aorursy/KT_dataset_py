
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.shape,test.shape
train.head(6)
test.head(6)
train.dtypes
train.columns[train.isnull().any()], test.columns[test.isnull().any()]
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data
total=test.isnull().sum().sort_values(ascending=False)
percent=(test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data
train=train.drop(['Ticket','Cabin'],axis=1)
test=test.drop(['Ticket','Cabin'],axis=1)
combine=[train,test]
for dataset in combine:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
pd.crosstab(train['Title'],train['Sex'])
for dataset in combine:
    dataset['Title']=dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train[['Title','Survived']].groupby(['Title'],as_index=False).mean()
title_mapping={'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}
for dataset in combine:
    dataset['Title']=dataset['Title'].map(title_mapping)
    dataset['Title']=dataset['Title'].fillna(0)
       
train.head(6)
    
train['Age']=train['Age'].fillna(train['Age'].median(skipna=True))
test['Age']=test['Age'].fillna(test['Age'].median(skipna=True))
train['Age']=train['Age'].astype(int)
test['Age']=test['Age'].astype(int)
train['AgeBand']=pd.cut(train['Age'],5)
train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)
combine=[train,test]
for dataset in combine:
    dataset.loc[dataset['Age']<=16,'Age']=0
    dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1
    dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2
    dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=3
    dataset.loc[(dataset['Age']>64),'Age']=4
    
train.head(6)
train=train.drop(['AgeBand'],axis=1)
combine=[train,test]
for dataset in combine:
    dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
    
train[['FamilySize','Survived']].groupby(['FamilySize'],as_index=False).mean().sort_values(by='Survived',ascending=False)
    
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train = train.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test = test.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train, test]

train.head()
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(train['Embarked'].dropna().mode()[0])
    
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()
train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
combine = [train, test]
    
train.head(10)
for dataset in combine:
    dataset['Sex']=dataset['Sex'].map({'female':1,'male':0}).astype(int)
train.head(10)
plt.figure(figsize=(32,32)) 
sns.heatmap(train.corr(),vmin=0, annot=True, cbar=True, cmap="RdYlGn")
train=train.drop(['Name','PassengerId'],axis=1)
test=test.drop(['Name'],axis=1)
X_train=train.drop(['Survived'],axis=1)
Y_train=train.Survived
X_test=test.drop('PassengerId',axis=1).copy()
from sklearn.cross_validation import KFold,cross_val_score
k_fold=KFold(len(Y_train),n_folds=10,shuffle=True,random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
#svc
svc=SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
#knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
#Niave bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred1 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN','Random Forest', 'Naive Bayes','Stochastic Gradient Decent', 'Linear SVC','Decision Tree'],
    'Score': [acc_svc, acc_knn,acc_random_forest, acc_gaussian, acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
submission=pd.read_csv('../input/gender_submission.csv')
submission['Survived']= Y_pred1
submission['PassengerId']=test['PassengerId']
pd.DataFrame(submission,columns=['PassengerId','Survived']).to_csv('randomforest.csv',index=False)

scoring = 'accuracy'
results = cross_val_score(random_forest, X_train, Y_train, cv=k_fold, n_jobs=1, scoring=scoring)
results
from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators": [40,50,60],
              "max_depth": [3, 5],
              "min_samples_split": [5, 10],
              "min_samples_leaf": [5, 6],
              "max_leaf_nodes": [10, 15],
              "min_weight_fraction_leaf": [0.1]}
param_grid
grid_search = GridSearchCV(random_forest, param_grid=param_grid,scoring='accuracy',cv=k_fold,n_jobs=-1)
grid_search.fit(X_train, Y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
from xgboost import XGBClassifier
xgb=XGBClassifier(max_depth=5, n_estimators=900, learning_rate=1,gamma=0,min_child_weight=2, reg_alpha=0.1,subsample=0.8,cv=k_fold)
xgb.fit(X_train,Y_train)
Y_pred = xgb.predict(X_test)
acc_xgb = round(xgb.score(X_train, Y_train) * 100, 2)
acc_xgb