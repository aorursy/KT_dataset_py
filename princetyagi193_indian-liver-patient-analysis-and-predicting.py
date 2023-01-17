# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

%matplotlib inline

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/indian_liver_patient.csv')
data.head()
data.isnull().sum()
data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean(),inplace=True)
data.describe(include='all')
data.columns
sns.countplot(label='count',x='Dataset',data=data);
sns.catplot(data=data,y='Age',x='Gender',hue='Dataset',jitter=0.4);
sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=data, kind="reg")
sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=data, kind="reg")
sns.jointplot("Total_Protiens", "Albumin", data=data, kind="reg")
data.corr()
data = pd.concat([data,pd.get_dummies(data['Gender'], prefix = 'Gender')], axis=1)
X = data.drop(['Gender','Dataset','Direct_Bilirubin','Aspartate_Aminotransferase'], axis=1)

y = data['Dataset']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
logistic=LogisticRegression()

logistic.fit(X_train,y_train)

logispredicted=logistic.predict(X_test)

print('Training Score:',logistic.score(X_train, y_train))

print('Testing Score:',logistic.score(X_test, y_test))

print('Accuracy:',accuracy_score(y_test,logispredicted))

print('Confusion Matrix: \n', confusion_matrix(y_test,logispredicted))
svmclf = svm.SVC(gamma='scale')

svmclf.fit(X_train,y_train)

svmpredicted=logistic.predict(X_test)

print('Training Score:',svmclf.score(X_train, y_train))

print('Testing Score:',svmclf.score(X_test, y_test))

print('Accuracy:',accuracy_score(y_test,svmpredicted))

print('Confusion Matrix: \n', confusion_matrix(y_test,svmpredicted))
# Random Forest



randomforest = RandomForestClassifier(n_estimators=100)

randomforest.fit(X_train, y_train)

#Predict Output

predicted = randomforest.predict(X_test)



print('Training Score:',randomforest.score(X_train, y_train))

print('Testing Score:',randomforest.score(X_test, y_test))

print('Accuracy:',accuracy_score(y_test,predicted))

print('Confusion Matrix: \n', confusion_matrix(y_test,predicted))
linear = linear_model.LinearRegression()

linear.fit(X_train, y_train)

#Predict Output

lpredicted = linear.predict(X_test)



print('Training Score:',linear.score(X_train, y_train))

print('Testing Score:',linear.score(X_test, y_test))

models = pd.DataFrame({

    'Model': [ 'Logistic Regression', 'SVM','Random Forest','Linear Regression'],

    'Score': [ logistic.score(X_train, y_train), svmclf.score(X_train, y_train), randomforest.score(X_train, y_train),linear.score(X_train, y_train)],

    'Test Score': [ logistic.score(X_test, y_test), svmclf.score(X_test, y_test), randomforest.score(X_test, y_test),linear.score(X_test, y_test)]})

models.sort_values(by='Test Score', ascending=False)