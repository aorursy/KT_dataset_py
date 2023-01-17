# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns 
dataset = pd.read_csv("../input/bank-additional-full.csv",sep=";")
dataset.head()
dataset.info()
dataset.describe()
dataset = dataset[['age','job','marital','loan','education','default','housing','y']]
dataset.tail()
dataset = pd.get_dummies(dataset,drop_first=True)
dataset.info()
X = dataset.iloc[:,0:28].values
y = dataset.iloc[:,28].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
print("Report:",classification_report(y_test,y_pred))
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred))
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10,n_jobs=-1)
accuracies.mean()
score=classifier.score(X_test,y_test)
plt.figure(figsize=(9,9));
sns.heatmap(cm,annot=True,fmt = ".3f",linewidth=.5,square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel("Predicted label");
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title,size=15);
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier1 = LogisticRegression()
classifier1.fit(X_train,y_train)
# Predicting the Test set results
y_pred1 = classifier1.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
print("Report:",classification_report(y_test,y_pred1))
print("Accuracy:",accuracy_score(y_test,y_pred1))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred1))
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies1 = cross_val_score(estimator = classifier1, X = X_train, y = y_train, cv = 10,n_jobs=-1)
accuracies1.mean()
score1=classifier1.score(X_test,y_test)
plt.figure(figsize=(9,9));
sns.heatmap(cm1,annot=True,fmt = ".3f",linewidth=.5,square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel("Predicted label");
all_sample_title = 'Accuracy Score: {0}'.format(score1)
plt.title(all_sample_title,size=15);
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
classifier2 = LogisticRegression()
classifier2.fit(X_train,y_train)
# Predicting the Test set results
y_pred2 = classifier2.predict(X_test)
cm2 = confusion_matrix(y_test, y_pred2)
print("Report:",classification_report(y_test,y_pred2))
print("Accuracy:",accuracy_score(y_test,y_pred2))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred2))
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies2 = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv = 10,n_jobs=-1)
accuracies2.mean()
score2=classifier2.score(X_test,y_test)
plt.figure(figsize=(9,9));
sns.heatmap(cm2,annot=True,fmt = ".3f",linewidth=.5,square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel("Predicted label");
all_sample_title = 'Accuracy Score: {0}'.format(score2)
plt.title(all_sample_title,size=15);
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'sigmoid', random_state = 0)
classifier3.fit(X_train, y_train)
# Predicting the Test set results
y_pred3 = classifier3.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test, y_pred3)
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies3 = cross_val_score(estimator = classifier3, X = X_train, y = y_train, cv = 10,n_jobs=-1)
accuracies3.mean()
print("Report:",classification_report(y_test,y_pred3))
print("Accuracy:",accuracy_score(y_test,y_pred3))
print("Confusion Matrix:",confusion_matrix(y_test,y_pred3))
score3=classifier3.score(X_test,y_test)
plt.figure(figsize=(9,9));
sns.heatmap(cm3,annot=True,fmt = ".3f",linewidth=.5,square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel("Predicted label");
all_sample_title = 'Accuracy Score: {0}'.format(score3)
plt.title(all_sample_title,size=15);