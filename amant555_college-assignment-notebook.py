import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
data=pd.read_csv("../input/Social_Network_Ads.csv")

data.head()
print("Total number of trainning elements: ",len(data))
def check_unique():

    for col in data.columns:

        print("Unique in ",col,len(data[col].unique()))

        print("NaN in ",col,data[col].isnull().values.any(),"\n")

check_unique()
data.info()
data=pd.get_dummies(data)

Y=data["Purchased"]

data=data.drop("Purchased",axis=1)

X=data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
#Polynomial Kernel

svclassifier = SVC(kernel='poly', degree=8)

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print("Accuracy Score ",accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
#Gaussian Kernel

svclassifier = SVC(kernel='rbf')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print("Accuracy Score ",accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
#Sigmoid Kernel

svclassifier = SVC(kernel='sigmoid')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print("Accuracy Score ",accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))