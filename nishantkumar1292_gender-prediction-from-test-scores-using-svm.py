import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns #for plotting
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import os
print(os.listdir("../input"))
#load data
data_file = "../input/StudentsPerformance.csv"
data = pd.read_csv(data_file)
data.head()
#relationship between scores
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0])
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1])
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2])
#relationship between scores (gender analysis)
f, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True)
sns.scatterplot(x="math score", y="reading score", data=data, ax=axes[0], hue="gender")
sns.scatterplot(x="reading score", y="writing score", data=data, ax=axes[1], hue="gender")
sns.scatterplot(x="writing score", y="math score", data=data, ax=axes[2], hue="gender")
#prepare X and y
X = data[['math score', 'writing score', 'reading score']]
scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
X = scaler.fit_transform(X)
y = data[['gender']]
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
#training
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)
#making predictions
y_pred = svclassifier.predict(X_test)
#evaluating the algorithm
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
#training with degree=2
from sklearn.svm import SVC  
svclassifier = SVC(kernel='poly', degree=2)  
svclassifier.fit(X_train, y_train)
#prediction
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
svclassifier = SVC(kernel='rbf')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))