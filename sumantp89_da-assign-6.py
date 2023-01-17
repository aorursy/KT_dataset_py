
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/Absenteeism_at_work.csv', delimiter=',')
features = df.iloc[:, 1:14].values
target = df.iloc[:, 14].values

df.head()
df.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3)
#SVM
svclassifier = SVC(kernel='linear',C=1.0,gamma=0.1)  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test) 
print("Accuracy for SVM Classifier: ", accuracy_score(y_test, y_pred) * 100)
#Decision Trees
dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Accuracy for Decision Trees: ", accuracy_score(y_test, y_pred_dt) * 100)
#Random Forest
rf = RandomForestClassifier(n_estimators=10, criterion='gini')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Accuracy for Random Forest: ", accuracy_score(y_test, y_pred_rf) * 100)
#KNN
accuracies = []
for i in range(1,50):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_knn))
print("Maximum Accuracy for KNN Classifier: {} with n_neighbors = {}".format(np.max(accuracies)*100, np.argmax(accuracies)+1))
