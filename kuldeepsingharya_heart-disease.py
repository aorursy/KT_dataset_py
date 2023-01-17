import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

%matplotlib inline

import os

print(os.listdir("../input"))

train=pd.read_csv('../input/heart.csv')
train.info()
train.shape
train.describe()
train.head()
# normaliztion

train=(train - np.min(train))/(np.max(train)-np.min(train)).values
train.head()
# lets define target value and attributes values

X=train[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]

Y=train["target"]

from sklearn.model_selection import train_test_split    # split data train & test

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)
# Applying Logistic regression algorithms

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,Y_train)

LR_model=model.score(X_test,Y_test)

print(LR_model)
print('Coefficient: \n', model.coef_)

print('Intercept: \n', model.intercept_)      
predictions=model.predict(X_test)

# classification report

from sklearn.metrics import classification_report

print(classification_report(Y_test,predictions))
# lets confusion matrix

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,predictions)

print(cm)

# Accuracy

from sklearn.metrics import accuracy_score

accuracy_score(Y_test,predictions)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

model_1=DecisionTreeClassifier()

model_1.fit(X_train,Y_train)

model_1_score=model_1.score(X_test,Y_test)

print(model_1_score)