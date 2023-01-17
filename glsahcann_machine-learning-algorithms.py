# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import data
data = pd.read_csv("../input/data.csv")
data.head()
data.info()
data.drop(["id","Unnamed: 32"],axis = 1,inplace = True)
data.head()
# binary classification
data.diagnosis.unique()
# list comprehention
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
data.diagnosis.unique()
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis = 1)
# normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3 , random_state = 1)
# svm
from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train,y_train)
# accuracy score
print("accuracy of svm algo:",svm.score(x_test,y_test))
# naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
# accuracy score
print("accuracy of naive bayes algo:",nb.score(x_test,y_test))
# decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
# accuracy score
print("accuracy of decision tree algo:",dt.score(x_test,y_test))
# random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,random_state = 1)
rf.fit(x_train,y_train)
# accuracy score
print("accuracy of random forest algo:",rf.score(x_test,y_test))
# confusion matrix
y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

import seaborn as sns
sns.heatmap(cm, annot = True, linewidths = 0.5 , linecolor = "yellow", fmt =".0f")
plt.xlabel("y_true")
plt.ylabel("y_pred")
plt.show()