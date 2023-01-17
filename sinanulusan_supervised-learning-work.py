# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()
data.info()
data.describe()
plt.subplots(figsize =(15,9))
sns.heatmap(data.corr(),cmap="Greens",annot = True)
plt.show()
sns.countplot(x = "target", data = data)
data.loc[:,"target"].value_counts()
##### from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
x_data,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']
x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size = 0.3, random_state =42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)

print(" Accuracy result for K = 3: ",knn.score(x_test,y_test))

# Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of K
for i, K in enumerate(neig):
    # K from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors = K)
    # Fit with KNN
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize = [13,8])
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
from sklearn.linear_model import LogisticRegression

# normalization
x = (x_data - np.min(x_data)/(np.max(x_data) - np.min(x_data)))
# train test split
from sklearn.model_selection import train_test_split
x_data,y = data.loc[:,data.columns != 'target'], data.loc[:,'target']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

lr = LogisticRegression(random_state = 42)
lr.fit(x_train,y_train)

print("Accuracy result of Logistic Regression is {} ".format(lr.score(x_test,y_test)))
from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

print("Accuracy result of SVM is : ",svm.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

print("Accuracy result of Naive Bayes is ",nb.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print("Accuracy result of Decision Tree Classifier is ",dt.score(x_test,y_test))

y_pred = dt.predict(x_test)
y_true = y_test

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred,y_true)

# we can visualization
f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(x_train,y_train)

print("Accuracy result of Random Forest Classification is ",rf.score(x_test,y_test))

y_pred = rf.predict(x_test)
y_true = y_test

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

# %% cm visualization
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm,annot = True,linewidths = 0.5,linecolor = "red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()