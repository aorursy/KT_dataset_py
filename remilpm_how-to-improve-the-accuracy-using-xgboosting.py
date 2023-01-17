# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn 

%matplotlib inline

import os

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import Imputer

#importing all the required ML packages

from sklearn.linear_model import LogisticRegression #logistic regression

from sklearn import svm #support vector Machine

from sklearn.ensemble import RandomForestClassifier #Random Forest

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.naive_bayes import GaussianNB #Naive bayes

from sklearn.tree import DecisionTreeClassifier #Decision Tree

from sklearn import metrics #accuracy measure

from sklearn.metrics import confusion_matrix #for confusion matrix

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import xgboost as xgb

from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
DB1=pd.read_csv("../input/diabetes/diabetes.csv")

DB1.head()
DX=DB1.copy()

DY=DB1.copy()

DX=DX.drop(['Outcome'], axis=1)

DX.head()

DY=DY['Outcome']

DY.head()

seed = 7

test_size = 0.33

DX_train, DX_test, Dy_train, Dy_test = train_test_split(DX, DY, test_size=test_size, random_state=seed)
model1=XGBClassifier()

model1.fit(DX_train,Dy_train)
Dy_pred=model1.predict(DX_test)

Dpredictions = [round(value) for value in Dy_pred]

Dpredictions
# evaluate predictions

Daccuracy = accuracy_score(Dy_test, Dpredictions)

print("Accuracy: %.2f%%" % (Daccuracy * 100.0))
### cross validation

X =DX

y = DB1['Outcome']

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

xyz=[]

accuracy=[]

std=[]

classifiers=['Naive Bayes','Linear Svm','Radial Svm','Logistic Regression','Decision Tree','KNN','Random Forest']

models=[GaussianNB(), svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier(),

        KNeighborsClassifier(n_neighbors=9),RandomForestClassifier(n_estimators=100)]

for i in models:

    model = i

    cv_result = cross_val_score(model,X,y, cv = kfold,scoring = "accuracy")

    cv_result=cv_result

    xyz.append(cv_result.mean())

    std.append(cv_result.std())

    accuracy.append(cv_result)

models_dataframe=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       

models_dataframe
#Comparing all predictions 

#=========================

#Naive Bayes           75.52%

#Linear Svm            77.21%

#Radial Svm            65.10%

#Logistic Regression   76.95%

#Decision Tree         69.26%

#KNN                   73.96%

#Random Forest         76.30%

#XGBoosting            77.95%



#XGBoosting is found to be more efficient