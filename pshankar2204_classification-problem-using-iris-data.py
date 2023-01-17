import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
iris=pd.read_csv("../input/Iris.csv")

iris.head()
iris['Species'].value_counts()
iris.describe()
#split the data set for train and test
X = iris.drop('Species',axis=1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import seaborn as sns
ax=sns.violinplot(iris['Species'],iris['SepalLengthCm'])
ax=sns.violinplot(iris['Species'],iris['SepalWidthCm'])
ax=sns.violinplot(iris['Species'],iris['PetalLengthCm'])
ax=sns.violinplot(iris['Species'],iris['PetalWidthCm'])
ax=sns.FacetGrid(iris, hue="Species", size=5) 
ax.map(plt.scatter, "SepalLengthCm", "SepalWidthCm") 
ax.add_legend()
#Logistic Regression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
#print(X_test[0:5])
#print(y_pred[0:5])
print(confusion_matrix(y_pred,y_test))
print('accuracy is',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
#Decision tree
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred =dtc.predict(X_test)
print('accuracy is',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
#naive bayes 
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
y_pred=nb.predict(X_test)
print('accuracy ',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
#Support Vector Machine
from sklearn.svm import LinearSVC
sv=LinearSVC()
sv.fit(X_train,y_train)
y_pred=sv.predict(X_test)
print('accuracy ',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print('accuracy ',accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))
print("logistc regression ")
print(cross_val_score(logreg,X_train,y_train,scoring='accuracy',cv=10).mean() *100)
print("decision tree classification ")
print(cross_val_score(dtc,X_train,y_train,scoring='accuracy',cv=10).mean() *100)
print("naive bayes ")
print(cross_val_score(nb,X_train,y_train,scoring='accuracy',cv=10).mean() *100)
print("support vector machine  ")
print(cross_val_score(sv,X_train,y_train,scoring='accuracy',cv=10).mean() *100)
print("k nearest neighbours ")
print(cross_val_score(knn,X_train,y_train,scoring='accuracy',cv=10).mean() *100)