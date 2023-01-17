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
data = pd.read_csv("/kaggle/input/zoo-animal-classification/zoo.csv")
data
#data.drop(["animal_name"],axis=1,inplace=True)
y = data.predator.values
x_data = data.drop(["predator"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Logistic Regression Accuracy: ",lr.score(x_test,y_test))
#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
print("K-nearest NeighborAccuracy:",knn.score(x_test,y_test))
#Support Vector MAchine
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)
print("Support Vector Machine Accuracy:",svm.score(x_test,y_test))
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("Naive Bayes Accuracy:",nb.score(x_test,y_test))
#decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("DecisionTree Accuracy:",dt.score(x_test,y_test))
#random forest Classification model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(x_train,y_train)
print("Random Forest Classification Accuracy:",rf.score(x_test,y_test))
#Confusion MAtrix model
y_true = y_test
y_predict = rf.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_predict)

import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(4,4))
sns.heatmap(cm,annot= True,linewidths = 0.5,linecolor = "red",fmt = ".0f",ax=ax)
plt.xlabel("y_predict")
plt.ylabel("y_true")
plt.show()