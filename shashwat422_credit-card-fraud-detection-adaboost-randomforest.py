import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#load dataset

dataset = pd.read_csv('../input/creditcard.csv')

dataset.shape
#plot histogram of each parameter

dataset.hist(figsize=(20,20))

plt.show()
#no of frauds in dataset

fraud = dataset[dataset['Class']==1]

valid = dataset[dataset['Class']==0]



outlier_fraction = len(fraud)/float(len(valid))

print(outlier_fraction)



print('fraud: {}'.format(len(fraud)))

print('valid: {}'.format(len(valid)))
corrmat = dataset.corr()

fig = plt.figure(figsize=(12,9))



sns.heatmap(corrmat, vmax = .8, square = True)

plt.show()
from sklearn.model_selection import train_test_split

X = dataset.iloc[:,:30]

Y = dataset.iloc[:,30]

print(X.shape)

print(Y.shape)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.25,random_state=0)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn import ensemble



# Creating classifier Object

ada = ensemble.AdaBoostClassifier()

#Fitting the classifier to training data

ada.fit(X_train,Y_train)



# Making Predictions

ada_pred = ada.predict(X_test)



print("Traing Score:%f"%ada.score(X_train,Y_train))

print("Testing Score:%f"%ada.score(X_test,Y_test))
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

print(accuracy_score(Y_test,ada_pred))

print(classification_report(Y_test,ada_pred))

print(confusion_matrix(Y_test,ada_pred))
from sklearn import ensemble



# Initializing Classifier Object

rf = ensemble.RandomForestClassifier()

# Fitting the classifier to training data

rf.fit(X_train,Y_train)



# Making Predictions

rf_pred = rf.predict(X_test)



print("Traing Score:%f"%rf.score(X_train,Y_train))

print("Testing Score:%f"%rf.score(X_test,Y_test))
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

print(accuracy_score(Y_test,rf_pred))

print(classification_report(Y_test,rf_pred))

print(confusion_matrix(Y_test,rf_pred))