# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.head()
data.info()
data.describe()
sns.countplot(x="diagnosis", data=data)

data.loc[:,'diagnosis'].value_counts()
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# malignant = M  kotu huylu tumor

# benign = B     iyi huylu tumor

M = data[data.diagnosis == "M"]

B = data[data.diagnosis == "B"]

#Scatter Plot

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Bad",alpha= 0.4)

plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Good",alpha= 0.4)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend()

plt.show()
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)
# normalization 

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)

# Naive bayes 

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))

 
x.shape
y
#CONFUSİON MATRİX FOR NAIVE BAYES

y_pred = nb.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)
f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
#  train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)
print("Decision Tree algorithm accuracy: ", dt.score(x_test,y_test))
#CONFUSİON MATRİX FOR DESICION TREE

y_pred = dt.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)

print("Random Forest algorithm accuracy: ",rf.score(x_test,y_test))
#CONFUSİON MATRİX FOR RANDOM FOREST

y_pred = rf.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)
f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

svm = SVC(random_state = 42)

svm.fit(x_train,y_train)
print("SVM algorithm accuracy: ",svm.score(x_test,y_test))
#CONFUSİON MATRİX FOR SVM

y_pred = svm.predict(x_test)

y_true = y_test

#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)
f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()