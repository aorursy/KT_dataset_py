# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# load dataset
train = pd.read_csv("../input/train.csv")

# drop columns 
train.drop(columns=['PassengerId','Cabin','Name','Ticket'],inplace=True)

# check if any null values
train.info()
# fill null var
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
# encode label
from sklearn.preprocessing import LabelEncoder
labelEncoder_X_1 = LabelEncoder()
train["Sex"] = labelEncoder_X_1.fit_transform(train["Sex"])
labelEncoder_X_2 = LabelEncoder()
train["Embarked"] = labelEncoder_X_2.fit_transform(train["Embarked"])
# get dummy var
obj_feature = ['Embarked']
train = pd.get_dummies(train,columns=obj_feature,drop_first=True)
# save feature and target
X = train.drop(columns="Survived").values
y = train["Survived"].values
# split dataset to train/test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.2,
                                                 random_state=0)
#feature scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
#-----------------------------------------------------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense

# initialization
classifier = Sequential()

# add layer
classifier.add(Dense(units = 5,
                     activation='relu',
                     kernel_initializer='uniform',
                     input_dim = 8))
# add layer
classifier.add(Dense(units = 5,
                     activation='relu',
                     kernel_initializer='uniform'))
# add layer
classifier.add(Dense(units = 1,
                     activation='sigmoid',
                     kernel_initializer='uniform'))
# compile
classifier.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
# fit model
classifier.fit(X_train_scaler,y_train,batch_size=10,epochs=100)

# predict on test 
y_pred_1 = classifier.predict(X_test_scaler)
y_pred_1 = (y_pred_1 > 0.5)

# get confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_1)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_2 = LogisticRegression()
classifier_2.fit(X_train_scaler,y_train)

# predict on test 
y_pred_2 = classifier_2.predict(X_test_scaler)

# get confusion matrix
cm_2 = confusion_matrix(y_test,y_pred_2)

# cross validation
from sklearn.model_selection import cross_val_score
accuracies_2 = cross_val_score(classifier_2,X = X_train_scaler,y = y_train,cv=10)
from sklearn.svm import SVC
classifier_3 = SVC(kernel = 'rbf',random_state=0,gamma=0.1)
classifier_3.fit(X_train_scaler,y_train)

# predict on test 
y_pred_3 = classifier_3.predict(X_test_scaler)

# get confusion matrix
from sklearn.metrics import confusion_matrix
cm_3 = confusion_matrix(y_test,y_pred_3)

# cross validation
from sklearn.model_selection import cross_val_score
accuracies_3 = cross_val_score(classifier_3,X = X_train_scaler,y = y_train,cv=10)

# Applying Grid Search to find the best model and best parameters
"""
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1],'kernel':['rbf'],'gamma':[0.1]}]
grid_search = GridSearchCV(estimator = classifier_3,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train_scaler,y_train)
best_acc = grid_search.best_score_
best_para = grid_search.best_params_
"""
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier_4 = GaussianNB()
classifier_4.fit(X_train_scaler,y_train)

# predict on test 
y_pred_4 = classifier_4.predict(X_test_scaler)

# get confusion matrix
cm_4 = confusion_matrix(y_test,y_pred_4)

# cross validation
accuracies_4 = cross_val_score(classifier_4,X = X_train_scaler,y = y_train,cv=10)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier_5 = RandomForestClassifier(n_estimators = 1000,criterion = 'entropy',random_state=0)
classifier_5.fit(X_train_scaler,y_train)

# predict on test 
y_pred_5 = classifier_5.predict(X_test_scaler)

# get confusion matrix
cm_5 = confusion_matrix(y_test,y_pred_5)

# cross validation
accuracies_5 = cross_val_score(classifier_5,X = X_train_scaler,y = y_train,cv=10)
from xgboost import XGBClassifier
classifier_6 = XGBClassifier(gamma=0.7,
                             learning_rate=0.18,
                             max_delta_step = 0,
                             max_depth=3,
                             min_child_weight = 1.2,
                             n_estimators=150,
                             scale_pos_weight=1.1)
classifier_6.fit(X_train_scaler,y_train)

# predict on test 
y_pred_6 = classifier_6.predict(X_test_scaler)

# get confusion matrix
cm_6 = confusion_matrix(y_test,y_pred_6)

# cross validation
accuracies_6 = cross_val_score(classifier_6,X = X_train_scaler,y = y_train,cv=10)

# Applying Grid Search to find the best model and best parameters
"""
from sklearn.model_selection import GridSearchCV
parameters_xgb = [{'max_depth':[3],
                   'learning_rate':[0.18],
                   'n_estimators':[150],
                   'gamma':[0.7],
                   'min_child_weight':[1.1,1.2,1.3,1.4,1.5,1.6],
                   'max_delta_step':[0,0.1,0.2,0.3,0.4],
                   'scale_pos_weight':[0.8,0.9,1,1.1,1.2]}]
grid_search_xgb = GridSearchCV(estimator = classifier_6,
                           param_grid = parameters_xgb,
                           scoring = 'accuracy',
                           cv = 10)
grid_search_xgb = grid_search_xgb.fit(X_train_scaler,y_train)
best_acc_xgb = grid_search_xgb.best_score_
best_para_xgb = grid_search_xgb.best_params_
"""
print ("Logistic Regression:",np.mean(accuracies_2),np.std(accuracies_2))
print ("Kernel SVM:",np.mean(accuracies_3),np.std(accuracies_3))
print ("Naive Bayes:",np.mean(accuracies_4),np.std(accuracies_4))
print ("RandomForestClassifier:",np.mean(accuracies_5),np.std(accuracies_5))
print ("XGBClassifier:",np.mean(accuracies_6),np.std(accuracies_6))
test = pd.read_csv("../input/test.csv")
test_id = test['PassengerId']
test.drop(columns=['PassengerId','Cabin','Name','Ticket'],inplace=True)

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

labelEncoder_X_1 = LabelEncoder()
test["Sex"] = labelEncoder_X_1.fit_transform(test["Sex"])
labelEncoder_X_2 = LabelEncoder()
test["Embarked"] = labelEncoder_X_2.fit_transform(test["Embarked"])

#get dummy var
test = pd.get_dummies(test,columns=obj_feature,drop_first=True)

#select features
X = test.values

# scaler - trans
test_X_scaler = scaler.transform(X)

# Apply model on test dataset

# KSVM
test_pred_3 = classifier_3.predict(test_X_scaler)
submission_3 = pd.DataFrame({'PassengerId':test_id.values,'Survived':test_pred_3})
#submission_3.to_csv("submission0925_ksvm.csv",index=False)