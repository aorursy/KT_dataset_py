import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

print("lybraries are imported..!")
data=pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
print("File read successfully!")
data.head()
data.tail()
data.info()
data.shape
data.describe()
data.groupby('species').size()
data.groupby('species').mean()
data.hist(edgecolor='black', linewidth=1, figsize=(15,5))
sbn.pairplot(data,hue="species")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print("imported..!")
X=data.iloc[:, :-1].values
y=data.iloc[:,-1].values
print("Data seperated..!")
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print("Data splited into train set & test set..!")
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
print('Accuracy is: ', accuracy_score(y_pred,y_test))
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
print("classification using Naive Byes")
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
print("Accuracy Score:", accuracy_score(y_pred,y_test))
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print("classification using Support Vector Machine's")
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
