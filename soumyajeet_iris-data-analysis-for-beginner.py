import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#importing the datset
dataset = pd.read_csv('../input/Iris.csv')
dataset.head()
# ID is unrelevat collumn , so i want to drop it .
dataset =dataset.drop("Id",axis=1)
dataset.head()
#finding is there any null value in data set or not
dataset.isnull().values.any()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)
#for y converting categorical data , for the i use get_dummies method , you can use lable encoding an one hot encoding 
y=pd.get_dummies(y,columns=['Species'])
print (y)
# droping one dummy column 
y = y.iloc[:, 1:3].values
print (y)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print (X_test)
print(y_test)
print(X_train)
print(y_train)
X.shape
X_train.shape
X_test.shape

y.shape
y_test.shape
y_train.shape
#i am using k-nearest Neighbours over here.
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print (y_pred)
print(y_test)
# Summary of the predictions made by the classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test.argmax(axis=1),y_pred.argmax(axis=1)))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
