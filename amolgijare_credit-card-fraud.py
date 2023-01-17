# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/creditcard.csv')
dataset.head()
X = dataset.iloc[:,1:30].values
y = dataset['Class'].values
class_counts = np.unique(y, return_counts = True)
classes = class_counts[0]
print(classes)
counts = class_counts[1]
print(counts)
plt.bar(classes, counts)
plt.xticks(classes, ('No Fraud','Fraud'))
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.show()
plt.scatter(dataset['Time'].values, dataset['Amount'].values)
plt.xlabel('Time')
plt.ylabel('Amount')
plt.show()
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0, n_jobs = -1)
lr_classifier.fit(X_train, y_train)
print(lr_classifier.coef_)
print(lr_classifier.intercept_)
print(lr_classifier.n_iter_)
y_pred = lr_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Logistic Regression")
print("True Possitive, False Positive \nFalse Negative, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_jobs = -1)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("K Nearest Neighbors Classifier")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 500, max_features=28, 
                                       criterion='entropy', n_jobs=-1, random_state = 100)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Random Forest Classifier")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion='entropy')
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Decision tree classifier")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
from sklearn.svm import SVC
sc_classifier = SVC(cache_size = 4000)
sc_classifier.fit(X_train, y_train)
y_pred = sc_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Support Vector Machine Classifier")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
X_train_no_fraud = []
y_train_no_fraud = []

X_train_fraud = []
y_train_fraud = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        X_train_no_fraud.append(X_train[i])
        y_train_no_fraud.append(y_train[i])
    else:
        X_train_fraud.append(X_train[i])
        y_train_fraud.append(y_train[i])
X_train_no_fraud = np.array(X_train_no_fraud)
y_train_no_fraud = np.array(y_train_no_fraud)

X_train_fraud = np.array(X_train_fraud)
y_train_fraud = np.array(y_train_fraud)
from sklearn.ensemble import IsolationForest
if_classifier = IsolationForest(n_estimators = 500, n_jobs=-1, random_state = 100)
if_classifier.fit(X_train_no_fraud)
y_pred = if_classifier.predict(X_train_fraud)
y_pred[y_pred > 0] = 0
y_pred[y_pred < 0] = 1

class_counts = np.unique(y_pred, return_counts = True)
classes = class_counts[0]
counts = class_counts[1]
plt.bar(classes, counts)
plt.xticks(classes, ('No Fraud','Fraud'))
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.show()
cm = confusion_matrix(y_train_fraud, y_pred)
print("Isolation forest")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_train_fraud, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_train_fraud, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
from sklearn.neighbors import LocalOutlierFactor
lof_classifier = LocalOutlierFactor(contamination = 0.002, n_jobs = -1)
y_pred = lof_classifier.fit_predict(X_test)
scores_pred = lof_classifier.negative_outlier_factor_
y_pred[y_pred > 0] = 0
y_pred[y_pred < 0] = 1

class_counts = np.unique(y_pred, return_counts = True)
classes = class_counts[0]
counts = class_counts[1]
plt.bar(classes, counts)
plt.xticks(classes, ('No Fraud','Fraud'))
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.show()
cm = confusion_matrix(y_test, y_pred)
print("Local Outlier factor")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
from sklearn.svm import OneClassSVM
oneclassSVM_classifier = OneClassSVM(cache_size = 3000)
oneclassSVM_classifier.fit(X_train_no_fraud, y_train_no_fraud)
y_pred = oneclassSVM_classifier.predict(X_test)
y_pred[y_pred > 0] = 0
y_pred[y_pred < 0] = 1


cm = confusion_matrix(y_test, y_pred)
print("One SVM classifier")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
from sklearn.covariance import EllipticEnvelope
ee_classifier = EllipticEnvelope(support_fraction = 10.0, random_state = 100)
ee_classifier.fit(X_train_no_fraud, y_train_no_fraud)
y_pred = ee_classifier.predict(X_test)
y_pred[y_pred > 0] = 0
y_pred[y_pred < 0] = 1


cm = confusion_matrix(y_test, y_pred)
print("Elliptical envelop classifier")
print("True Possitive, False Negative \nFalse Positive, True Negative \n Confusion Matrix = \n", cm)
ac = accuracy_score(y_test, y_pred)
print("\n\n Accuracy Score = True Positive + True Negative / (Total records) \n", ac)
print("\n\n", classification_report(y_test, y_pred))
false_pos_percentage = cm[1][0] * 100 / (cm[1][0] + cm[1][1])
print("\n\n false positive percentage: ", false_pos_percentage)
