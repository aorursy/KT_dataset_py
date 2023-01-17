import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
data = pd.read_csv("../input/bank-marketing/bank-additional-full.csv",sep=';')

data.head()
data.info()
newdata= data.drop(["job","education","default","contact","month","day_of_week","poutcome"],axis=1)

newdata["marital"] = [1 if each == "married" else 0 if each == "single" else 0.5 for each in newdata.marital]

newdata["housing"] = [0 if each== "no" else 1 for each in newdata.housing]

newdata["loan"] = [0 if each == "no" else 1 for each in newdata.loan]

newdata["y"] = [0 if each == "no" else 1 for each in newdata.y]

# Defining to y and x_data values for train data

y = newdata.y.values

x_data = newdata.drop(["y"],axis=1)
# normalization

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=0.28,random_state=0)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='entropy') #gini parametresi

dt.fit(x_train,y_train)

KararAgaci=dt.score(x_test,y_test)

print("Decison Tree Accuracy Score: ",KararAgaci)





y_predDT=dt.predict(x_test)



from sklearn.metrics import confusion_matrix,accuracy_score,f1_score, precision_score, recall_score, classification_report

cm=confusion_matrix(y_test,y_predDT)

print(cm)



print("Reports\n",classification_report(y_test,y_predDT))
# Random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50,criterion='entropy')

rf.fit(x_train, y_train)

print("Random Forest : ",rf.score(x_test,y_test))





y_predRF=rf.predict(x_test)

cmRF=confusion_matrix(y_test,y_predRF)

print(cmRF)





print("Reports\n",classification_report(y_test,y_predRF))
# Naive Byes 

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("Naive Bayes : ",nb.score(x_test,y_test))







y_predNB=nb.predict(x_test)

cmNB=confusion_matrix(y_test,y_predNB)

print(cmNB)

print("Reports\n",classification_report(y_test,y_predNB))
# SVM model

from sklearn.svm import SVC

svm = SVC(gamma='auto')

svm.fit(x_train,y_train)

print("SVM : ",svm.score(x_test,y_test))



y_predSVM=svm.predict(x_test)

cmSVM=confusion_matrix(y_test,y_predSVM)

print(cmSVM)

print("Reports\n",classification_report(y_test,y_predSVM))
# KNN model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4, metric='minkowski') #n_neighbors = k

knn.fit(x_train,y_train)

#prediction = knn.predict(x_test)

print("KNN :",knn.score(x_test,y_test))



y_predKNN=knn.predict(x_test)

cmKNN=confusion_matrix(y_test,y_predKNN)

print(cmKNN)

print("Reports\n",classification_report(y_test,y_predKNN))