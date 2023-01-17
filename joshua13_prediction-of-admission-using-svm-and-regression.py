import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

%matplotlib inline  

print(os.listdir("../input"))
df = pd.read_csv(r'../input/Admission_Predict.csv')
dataset =  np.array(df.values)

dataset = dataset[1:]

np.random.shuffle(dataset)

print(dataset)

X = df.iloc[:, :-1]

y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
clf = svm.SVR(gamma='scale', C=30, epsilon=0.0002,tol=0.001)

clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)

from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

clf.score(X_test,y_test)
reg = LinearRegression()

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

reg.score(X_test,y_test)