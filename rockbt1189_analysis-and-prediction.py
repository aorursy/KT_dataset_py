import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print(train.shape)

print(test.shape)

print(train.head(4))

print(test.head(4))
# found some NAN values,lets check 

print("In the train Dataset the no. of missing/NAN values in the columns are \n",train.isnull().sum())

print("\n In the test Dataset the no. of missing/NAN values in the columns are \n",test.isnull().sum())
#lets check the missing value perecentage in the dataset

print("Total missing value the train and test detaset are:",(train.isnull().sum().sum()+test.isnull().

                                                             sum().sum()))

print("In the train dataset missing value percentage of the veriables are:\n",

      train.isnull().sum()*100/len(train))



print("In the test dataset missing value percentage of the veriables are:\n",

      test.isnull().sum()*100/len(test))
#Seperate the Survived veriabler then we will merge and drop the dataset

Survived = train[["Survived"]]

train["source"] = "train"

test["source"] = "test"

dataset = pd.concat([train,test],axis=0)

print(dataset.shape)

print(dataset.head())

print(dataset.tail())

# drop the not required columns from the datasert

dataset.drop(["Cabin","Ticket","Survived","Name"],axis=1,inplace=True)

print(dataset.shape)

print("\n datatypes:\n")

dataset.info()
dataset["Parch"] = dataset["Parch"].astype("object")

dataset["Pclass"] = dataset["Pclass"].astype("object")

dataset["SibSp"] = dataset["SibSp"].astype("object")

dataset.info()
#let's check how is dataset veriables are destributed and what we can do for Age missing values.

# Basic descriptive analysis on Age Veriable to know the distribution

print(dataset["Age"].describe())

plt.rcParams["figure.figsize"]=12,4

sns.boxplot("Age",data=dataset)

plt.title("Age distribution Boxplot")

print("Missing value in Age:",dataset["Age"].fillna(30, inplace = True))

print(dataset["Age"].isnull().sum())
# We have also seen some missing valus in Embarked and Fare veriables .

#Because of the missing values are very less we can put mode for Embarked and mean for Fare.

print("\n Embarked value count:\n" ,dataset["Embarked"].value_counts())

print("Missing value in Embarked:",dataset["Embarked"].fillna("S", inplace = True))

print("mean value of Fare is ",dataset["Fare"].mean())

print("Missing value in Fare:",dataset["Fare"].fillna(dataset["Fare"].mean(), inplace = True))

print("Total Missing Values in the dataset is:",dataset.isnull().sum().sum())
train= dataset[dataset["source"]=="train"]

test= dataset[dataset["source"]=="test"]

train.drop("source",axis=1,inplace=True)

test.drop("source",axis=1,inplace=True)

train["Survived"] = Survived["Survived"]

print(train.shape)

print(test.shape)
#univariate data visualization for Numeric Data "Age","Fare","Survived":

plt.rcParams["figure.figsize"]=13,16



plt.subplot(3,2,1)

sns.countplot(train.Pclass)

plt.title("Pclass Frequency Distribution")



plt.subplot(3,2,2)

sns.countplot(train.Sex)

plt.title("Sex Frequency Distribution")



plt.subplot(3,2,3)

sns.distplot(train.Fare)

plt.title("Fare Frequency Distribution")

plt.subplot(3,2,4)

sns.boxplot(train.Fare)

plt.title("Fare Boxplot Distribution")



plt.subplot(3,2,5)

sns.countplot(train.Embarked)

plt.title("Embarked Frequency Distribution")



plt.subplot(3,2,6)

sns.countplot(train.Survived)

plt.title("Survived Frequency Distribution")
plt.rcParams["figure.figsize"]=13,16



plt.subplot(3,2,1)

sns.countplot(train.Pclass,hue=train["Survived"])

plt.title("Pclass Frequency Distribution")



plt.subplot(3,2,2)

sns.countplot(train.Sex,hue=train["Survived"])

plt.title("Pclass Frequency Distribution")



plt.subplot(3,2,3)

sns.boxplot(x ="Survived",y="Fare",data=train,hue="Sex")

plt.title("Fare Frequency Distribution")



plt.subplot(3,2,4)

sns.countplot(train.Embarked,hue=train["Survived"])

plt.title("Embarked Frequency Distribution")



plt.subplot(3,2,5)

sns.boxplot(x ="Pclass",y="Fare",data=train,hue="Survived")

plt.title("Pcalss Vs Fare")

plt.subplot(3,2,6)

sns.boxplot(x ="Parch",y="Fare",data=train,hue="Survived")

plt.title("Pcalss Vs Fare")

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn import metrics



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
X = train[["Age","Fare","Pclass","SibSp"]].values

Y = train[["Survived"]].values

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=120)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(xtrain,ytrain)

y_pred = LR.predict(xtest)

LR.score(xtrain, ytrain)
# Support Vector Machines

svc = SVC()

svc.fit(xtrain, ytrain)

y_pred = svc.predict(xtest)

svc.score(xtrain, ytrain)
#library

from sklearn.model_selection import cross_val_score

svc = SVC()

svm_accuracy = cross_val_score(estimator=svc,X=xtrain,y=ytrain,cv=5)

print("SVM Accuracy:",svm_accuracy)

print("Accuracy Mean:",svm_accuracy.mean()*100)

print("Accuracy STD:",svm_accuracy.std()*100)
'''svc = SVC()

#from sklearn import svm, grid_search

from sklearn.model_selection import GridSearchCV

grid_param = {'kernel':('linear', 'rbf','sigmoid'),

              'C':(1,0.25,0.5,0.75),

              'gamma': (0.001, 0.01, 0.1, 1),

              'shrinking':(True,False)

              }

gd_sr = GridSearchCV(estimator=svc,  

                     param_grid=grid_param,

                     scoring='accuracy',

                     cv=5,

                     n_jobs=-1)



gd_sr.fit(xtrain, ytrain) 

best_parameters = gd_sr.best_params_  

print('Best Parameters',best_parameters) 

svm_accuracy = gd_sr.best_score_  

print("Model Accuracy:",svm_accuracy)'''
#------------------CV and Tunning with Random Forest---------------

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=300, random_state=0)  

RFC_Accuracy = cross_val_score(estimator=RFC, X=xtrain, y=ytrain, cv=5)



print("RFC Accuracy:",RFC_Accuracy)

print("Accuracy Mean:",RFC_Accuracy.mean()*100)

print("Accuracy STD:",RFC_Accuracy.std()*100)
grid_param = {  

    'n_estimators': [30,50,100],

    'criterion': ['gini', 'entropy'],

    'bootstrap': [True, False],

    'max_features':["auto","sqrt"],

    'max_depth': [2,3,5],

    'max_leaf_nodes': [20,30,40]

}

gd_sr = GridSearchCV(estimator=RFC,  

                     param_grid=grid_param,

                     scoring='accuracy',

                     cv=5,

                     n_jobs=-1)

gd_sr.fit(xtrain,ytrain)

best_parameters = gd_sr.best_params_  

print(best_parameters)

best_result = gd_sr.best_score_  

print(best_result)
#basic xgboost

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

XGB = XGBClassifier()

XGB.fit(xtrain, ytrain)

y_pred = XGB.predict(xtest)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(ytest, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''#xgboost with Tuning

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

#basic xgboost

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier'''



'''cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}

ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 

             'objective': 'binary:logistic'}

optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 

                            cv_params, 

                             scoring = 'accuracy', cv = 5, n_jobs = -1) 

optimized_GBM.fit(xtrain, ytrain)

optimized_GBM.cv_results_'''



'''cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

ind_params = {'n_estimators': 1000, 'seed':0, 'colsample_bytree': 0.8, 

             'objective': 'binary:logistic', 'max_depth': 3, 'min_child_weight': 1}





optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 

                            cv_params, 

                             scoring = 'accuracy', cv = 5, n_jobs = -1)

optimized_GBM.fit(xtrain, ytrain)

optimized_GBM.cv_results_'''
# knn = KNeighborsClassifier(n_neighbors = 3)



# knn.fit(X_train, Y_train)



# Y_pred = knn.predict(X_test)



# knn.score(X_train, Y_train)



# Gaussian Naive Bayes



# gaussian = GaussianNB()



# gaussian.fit(X_train, Y_train)



# Y_pred = gaussian.predict(X_test)



# gaussian.score(X_train, Y_train)
#------------------------------------------------------

F_test = test[["Age","Fare","Pclass","SibSp"]].values

y_pred = XGB.predict(F_test)

#------------------------------------------------------

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('titanic.csv', index=False)
y_pred