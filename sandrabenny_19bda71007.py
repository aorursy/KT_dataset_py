import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import sklearn.model_selection as me

import sklearn.preprocessing as pre

import sklearn.linear_model as lm

import sklearn.metrics as ms



%matplotlib inline
df=pd.read_csv("../input/bda-2019-ml-test/Train_Mask.csv")
Y=df.iloc[:,1]
Y
corr=df.corr()
X=df.drop("flag",axis=1)
X
#splitting the dataset into train and test

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)
#KNN algorithm

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)

knn.fit(X_train, Y_train)
# Decision tree AlgorithmDecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)

dectree.fit(X_train, Y_train)
#Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(n_estimators = 60, criterion = 'entropy', random_state = 42)

ranfor.fit(X_train, Y_train)
#Random Forest Algorithm

from sklearn.ensemble import RandomForestClassifier

ranfor = RandomForestClassifier(n_estimators = 60, criterion = 'entropy', random_state = 42)

ranfor.fit(X_train, Y_train)
#Naive Bayes Algorithm

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, Y_train)
#Logistic Regression Algorithm

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)
# Making predictions on test dataset

Y_pred_logreg = logreg.predict(X_test)

Y_pred_knn = knn.predict(X_test)

Y_pred_nb = nb.predict(X_test)

Y_pred_dectree = dectree.predict(X_test)

Y_pred_ranfor = ranfor.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)

accuracy_knn = accuracy_score(Y_test, Y_pred_knn)

accuracy_nb = accuracy_score(Y_test, Y_pred_nb)

accuracy_dectree = accuracy_score(Y_test, Y_pred_dectree)

accuracy_ranfor = accuracy_score(Y_test, Y_pred_ranfor)
print("Logistic Regression: " + str(accuracy_logreg * 100))

print("K Nearest neighbors: " + str(accuracy_knn * 100))

print("Naive Bayes: " + str(accuracy_nb * 100))

print("Decision tree: " + str(accuracy_dectree * 100))

print("Random Forest: " + str(accuracy_ranfor * 100))
Test=pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")

Sample=pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
x_Test=Test.iloc[:,1]
Y_pred = ranfor.predict(Test)

Y_pred
Sample["flag"]=Y_pred
Sample.to_csv("submit_15.csv",index=False)
# Classification report

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred_ranfor))