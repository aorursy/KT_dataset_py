#importing modules

import numpy as np 

import pandas as pd 



#loading datasets

data = pd.read_csv("../input/data.csv")
#the fist 5 rows

data.head(5)
#the dataset summary

data.info()
#deleting useless columns

#deleting the "id" column

data.drop("id",axis=1,inplace=True)

#deleting the "Unnamed: 32" column

data.drop("Unnamed: 32",axis=1,inplace=True) 
#the result

data.info()
#the first 5 rows

data.head(5)
#counting the diagnosis variable

data.diagnosis.value_counts()
#diagnosis variable is a responsible variable for the classification

#replacing M and B with 1 and 0 respectively

data.diagnosis=data.diagnosis.map({'M':1,'B':0})
#counting the diagnosis variable

data.diagnosis.value_counts()
#finished the dataset preprocessing

#splitting dataset into training one and testing one

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.3,random_state=1234)
#finding out the results

print(train.shape)

print(test.shape)
#making independent variables for training

train_X = train.iloc[:, 1:31]

#making responsible variables for training

train_y=train.diagnosis

#making independent variables for testing

test_X= test.iloc[:, 1:31]

#making responsible variables for testing

test_y =test.diagnosis
#finding out the results

print(train_X.shape)

print(train_y.shape)

print(test_X.shape)

print(test_y.shape)
#Without Hyper Parameters Tuning

#1-1,DesicionTree

#1-2,Randomforest

#1-3,SVM

#1-4,kNearestNeighbors

#With Hyper Parameters Tuning

#2-1,DesicionTree

#2-2,Randomforest

#2-3,SVM

#2-4,kNearestNeighbors
#Without Hyper Parameters Tuning

#1-1,DesicionTree

#importing module

from sklearn.tree import DecisionTreeClassifier

#making the instance

model= DecisionTreeClassifier(random_state=1234)

#learning

model.fit(train_X,train_y)

#Prediction

prediction=model.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#Without Hyper Parameters Tuning

#1-2,Randomforest

#importing module

from sklearn.ensemble import RandomForestClassifier

#making the instance

model=RandomForestClassifier(n_jobs=-1,random_state=123)

#learning

model.fit(train_X,train_y)

#Prediction

prediction=model.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#Without Hyper Parameters Tuning

#1-3,SVM

#importing module

from sklearn import svm

#making the instance

model = svm.SVC(random_state=123)

#learning

model.fit(train_X,train_y)

#Prediction

prediction=model.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#Without Hyper Parameters Tuning

#1-4,kNearestNeighbors

#importing module

from sklearn.neighbors import KNeighborsClassifier

#making the instance

model = KNeighborsClassifier(n_jobs=-1)

#learning

model.fit(train_X,train_y)

#Prediction

prediction=model.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#With Hyper Parameters Tuning

#2-1,DesicionTree

#importing modules

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

#making the instance

model= DecisionTreeClassifier(random_state=1234)

#Hyper Parameters Set

params = {'max_features': ['auto', 'sqrt', 'log2'],

          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 

          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],

          'random_state':[123]}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

#Learning

model1.fit(train_X,train_y)

#The best hyper parameters set

print("Best Hyper Parameters:",model1.best_params_)

#Prediction

prediction=model1.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#With Hyper Parameters Tuning

#2-2,Randomforest

#importing modules

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

#making the instance

model=RandomForestClassifier()

#hyper parameters set

params = {'criterion':['gini','entropy'],

          'n_estimators':[10,15,20,25,30],

          'min_samples_leaf':[1,2,3],

          'min_samples_split':[3,4,5,6,7], 

          'random_state':[123],

          'n_jobs':[-1]}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

#learning

model1.fit(train_X,train_y)

#The best hyper parameters set

print("Best Hyper Parameters:\n",model1.best_params_)

#Prediction

prediction=model1.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#With Hyper Parameters Tuning

#2-3,SVM

#importing modules

from sklearn.model_selection import GridSearchCV

from sklearn import svm

#making the instance

model=svm.SVC()

#Hyper Parameters Set

params = {'C': [6,7,8,9,10,11,12], 

          'kernel': ['linear','rbf']}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)

#Learning

model1.fit(train_X,train_y)

#The best hyper parameters set

print("Best Hyper Parameters:\n",model1.best_params_)

#Prediction

prediction=model1.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#With Hyper Parameters Tuning

#2-4,kNearestNeighbors

#importing modules

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

#making the instance

model = KNeighborsClassifier(n_jobs=-1)

#Hyper Parameters Set

params = {'n_neighbors':[5,6,7,8,9,10],

          'leaf_size':[1,2,3,5],

          'weights':['uniform', 'distance'],

          'algorithm':['auto', 'ball_tree','kd_tree','brute'],

          'n_jobs':[-1]}

#Making models with hyper parameters sets

model1 = GridSearchCV(model, param_grid=params, n_jobs=1)

#Learning

model1.fit(train_X,train_y)

#The best hyper parameters set

print("Best Hyper Parameters:\n",model1.best_params_)

#Prediction

prediction=model1.predict(test_X)

#importing the metrics module

from sklearn import metrics

#evaluation(Accuracy)

print("Accuracy:",metrics.accuracy_score(prediction,test_y))

#evaluation(Confusion Metrix)

print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))
#Result           Without HyperParameterTuning      With HyperParameterTuning

#DecisionTree     0.929824561404                    0.912280701754   →not improved

#Randomforest     0.923976608187                    0.923976608187   →the same result

#SVM              0.614035087719                    0.93567251462    →dramatically improved

#kNearestNeighbor 0.93567251462                     0.93567251462    →the same result 



#The default hyper parameters set of DecisionTree, Randomforest and kNearestNeighbor looks not so bad.