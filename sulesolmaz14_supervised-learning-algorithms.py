# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data.tail()
#scatter plot
color_list = ['red' if i =='Abnormal' else 'green'  for i in data.loc[:, 'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'], c= color_list, figsize = [15,15], diagonal = 'hist', alpha=0.5, s =200, marker = '*', edgecolor = "black")
plt.show()

#N = data[data.class == "Normal"]
#A = data[data.class == "Abnormal"]
# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x, y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))

# train test split
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size = 0.3, random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3)
x, y = data.loc[:, data.columns != 'class'], data.loc[:, 'class']
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print('With KNN (K=3) accuracy is: {}', knn.score(x_test, y_test))# accuracy
#Model Complexity
neig = np.arange(1,25)
train_accuracy=[]
test_accuracy=[]

for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    #Fit with knn
    knn.fit(x_train, y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    #test accuracy
    test_accuracy.append(knn.score(x_test, y_test))
# Plot
plt.figure(figsize=[13, 8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
# SVM
from sklearn.svm import SVC

svm = SVC(random_state = 1)
svm.fit(x_train, y_train)
# TEST
print("Print Accuracy of SVM Algorithm:", svm.score(x_test, y_test))
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
# TEST
print("Print Accuracy of Naive Bayes Algorithm:", nb.score(x_test, y_test))
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
# TEST
print("Print Accuracy of Decision Tree Algorithm:", dt.score(x_test, y_test))
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
# TEST
print("Print Accuracy of Random Forest Algorithm:", rf.score(x_test, y_test))
y_pred = rf.predict(x_test)
y_true = y_test

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
#Visualisation
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm, annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
