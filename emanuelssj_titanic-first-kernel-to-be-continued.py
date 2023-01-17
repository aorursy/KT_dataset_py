# Core

import numpy as np 

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# Machine Learning and Hyperparameter Tuning

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



# Ignore warnings

import warnings

warnings.filterwarnings("ignore")



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



train_data.head()
features = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']



for i in range(len(features)):

    survived = train_data[train_data['Survived']==1][features[i]].value_counts()

    dead = train_data[train_data['Survived']==0][features[i]].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Dead']

    df.plot(kind='bar',stacked=True, figsize=(8,5))
features= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']



x = train_data[features]

y = train_data['Survived']



test_x = test_data[features]



x.head()
# Check Null values in features



x.isnull().sum()
# Fill Null Data



x['Age'] = x['Age'].fillna(x['Age'].median())

x['Embarked']= x['Embarked'].fillna(x['Embarked'].value_counts().index[0])



test_x['Age'] = test_x['Age'].fillna(test_x['Age'].median())

test_x['Fare'] = test_x['Fare'].fillna(test_x['Fare'].median())



x.isnull().sum()
# Encode Categorical Data



LE = LabelEncoder()



x['Sex'] = LE.fit_transform(x['Sex'])

x['Embarked'] = LE.fit_transform(x['Embarked'])



test_x['Sex'] = LE.fit_transform(test_x['Sex'])

test_x['Embarked'] = LE.fit_transform(test_x['Embarked'])



print(x.head())
# Check Null Values in target



y.isnull().sum()
# Split data



x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.1, random_state =0)
# Random Forest - Hyperparameter Tuning



RFClassifier = RandomForestClassifier()

n_estimators = range(10, 150)

## Search grid for optimal parameters

param_grid = {"n_estimators" : n_estimators}



model_rf = GridSearchCV(RFClassifier, param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



model_rf.fit(x_train,y_train)



# Best score

print(model_rf.best_score_)



#best estimator

model_rf.best_estimator_
# Random Forest



RFClassifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=66, n_jobs=None,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)

RFClassifier.fit(x_train,y_train)

RFClassifier.score(x_test,y_test)
# Decision Tree - Hyperparameter Tuning



DTClassifier = DecisionTreeClassifier()

min_samples_leaf = range(1, 1000)



## Search grid for optimal parameters

param_grid = {"min_samples_leaf" :min_samples_leaf}



model_dt = GridSearchCV(DTClassifier, param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



model_dt.fit(x_train,y_train)



# Best score

print(model_dt.best_score_)



#best estimator

model_dt.best_estimator_
# Decision Tree



DTClassifier = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

            max_features=None, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=10, min_samples_split=2,

            min_weight_fraction_leaf=0.0, presort=False, random_state=None,

            splitter='best')

DTClassifier.fit(x_train,y_train)

DTClassifier.score(x_test, y_test)
# knn - Hyperparameter Tuning



KNClassifier = KNeighborsClassifier()

n_neigh = range(10, 600)



## Search grid for optimal parameters

param_grid = {"n_neighbors" :n_neigh}



model_kn = GridSearchCV(KNClassifier, param_grid = param_grid, cv=5, scoring="accuracy", n_jobs= 4, verbose = 1)



model_kn.fit(x_train,y_train)



# Best score

print(model_kn.best_score_)



#best estimator

model_kn.best_estimator_
##knn



KNClassifier = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',

           metric_params=None, n_jobs=None, n_neighbors=13, p=2,

           weights='uniform')

KNClassifier.fit(x_train,y_train)

KNClassifier.score(x_test, y_test)
# XGBClassifier



xgbClassifier = XGBClassifier(colsample_bylevel= 0.9,

                    colsample_bytree = 0.8, 

                    gamma=0.99,

                    max_depth= 5,

                    min_child_weight= 1,

                    n_estimators= 10,

                    nthread= 4,

                    random_state= 2,

                    silent= True)

xgbClassifier.fit(x_train,y_train)

xgbClassifier.score(x_test,y_test)
# Support Vector Machines



SVCClassifier = SVC(gamma='auto')

SVCClassifier.fit(x_train,y_train)

SVCClassifier.score(x_test, y_test)
# Predict



prediction = xgbClassifier.predict(test_x)
# Submit

submission = pd.DataFrame({'PassengerId': test_data.PassengerId,

                           'Survived': prediction})

submission.to_csv('submission.csv', index=False)