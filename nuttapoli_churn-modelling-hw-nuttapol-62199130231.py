import os

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
dataFrame = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
dataFrame.shape
dataFrame.describe()
dataFrame.info()
X=dataFrame.loc[:,['CreditScore',

            'Age',

            'Tenure',

            'Balance',

            'NumOfProducts',

            'HasCrCard',

            'IsActiveMember',

            'EstimatedSalary']].values

y=dataFrame.iloc[:,-1].values
#Seaborn Plot

plt.figure(figsize=(16,16))

ax = sns.heatmap(dataFrame.corr(), linewidth=0.5, vmin=-1,

cmap='coolwarm', annot=True)

plt.title('Correlation heatmap')

plt.show()

#plt.savefig('myheatmap.png')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25,random_state =0)
from sklearn.preprocessing import StandardScaler

sc_X =StandardScaler()

X_train =sc_X.fit_transform(X_train)

X_test =sc_X.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
from sklearn.svm import SVC

classifier = SVC(kernel = 'poly', degree = 2, random_state = 0) #degree for non-linear

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)