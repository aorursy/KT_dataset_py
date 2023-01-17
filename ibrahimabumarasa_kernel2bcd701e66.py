import numpy as np

import pandas as pd 



#load existing dataset stored as csv file 

#dataset = pd.read_excel("../input/TestDataSet/newDataSet5.xlsx")



dataset = pd.read_csv("../input/datasetfinal/newDataSetFinal.csv")

## get columns' titles

print('Dataset contains on the following columns:')

titles = dataset.columns 

print(titles)



# Split Dataset into Data and target



print('\n\n*********Split Dataset into Data and target********************')



# first five rows

dsTarget = dataset['Category']

dsTrain = dataset[["ReceivedYear","Items","Byear","Location"]]

print("dsData \n",dsTrain.head())

print("dsTarget \n",dsTarget.head())

print('\n\n****************************************************************')
print('*********Split Dataset into 4 parts 2 for training and other for testing ********************')

print('\n*********Split dataset into 70% training set and 30% test set ********************')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dsTrain, dsTarget, test_size=0.3)

print('\n************Sample of train************\n',X_train.head())

print('\n************Sample of Test************\n' , y_test.head())
print('\n*********Apply Naive Bayes Model********************')

from sklearn.naive_bayes import GaussianNB as gnb

model = gnb()

model.fit(X_train, y_train)

y_pred=model.predict(X_test)



#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))

print("Classification Report:\n", metrics.classification_report(y_test,y_pred))
print('\n*********Apply Neural Network Model********************')

from sklearn.neural_network import MLPClassifier

na_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

na_model.fit(X_train, y_train)

na_pred=na_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, na_pred))

print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, na_pred))

print("Classification Report:\n", metrics.classification_report(y_test,na_pred))
print('\n*********Apply Decision Tree Model********************')

from sklearn.tree import DecisionTreeClassifier as dt

dt_model = dt(random_state=1)

dt_model.fit(X_train, y_train)

dt_pred=dt_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, dt_pred))

print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, dt_pred))

print("Classification Report:\n", metrics.classification_report(y_test,dt_pred))