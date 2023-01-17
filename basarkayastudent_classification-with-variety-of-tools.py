# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data
data.info()
data.target.value_counts()
x_data=data.drop(["target"], axis=1)

x=((x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))).values

y=data.target.values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train, y_train)

print("Logistic Regression Accuracy:", 100*lr.score(x_test, y_test), "%")
from sklearn.metrics import confusion_matrix

y_true=y_test

y_pred=lr.predict(x_test)

cmlr=confusion_matrix(y_true, y_pred)

f,ax=plt.subplots(figsize=(6,6))

sns.heatmap(cmlr, annot=True)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

score_list=[]

for each in range(1,30):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train, y_train)

    score_list.append(100*knn2.score(x_test, y_test))

    print("n=", each, "Accuracy:", 100*knn2.score(x_test, y_test), "%")
plt.plot([*range(1,30)], score_list)

plt.xlabel("n Value")

plt.ylabel("Accuracy %")

plt.show()
optimal_n_value=score_list.index(max(score_list))+1

knn=KNeighborsClassifier(n_neighbors=optimal_n_value)

knn.fit(x_train, y_train)

print("KNN Prediction Accuracy:", 100*knn.score(x_test, y_test), "%")
from sklearn.metrics import confusion_matrix

y_true=y_test

y_pred=knn.predict(x_test)

cmknn=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))

sns.heatmap(cmknn, annot=True)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.svm import SVC

svm=SVC(random_state=42)

svm.fit(x_train, y_train)

print("SVM Prediction Accuracy:", 100*svm.score(x_test, y_test), "%")
y_true=y_test

y_pred=svm.predict(x_test)

cmsvm=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))

sns.heatmap(cmsvm, annot=True)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train, y_train)

print("Naive Bayes Prediction Accuracy:", 100*nb.score(x_test, y_test), "%")
y_true=y_test

y_pred=nb.predict(x_test)

cmnb=confusion_matrix(y_true, y_pred)
f,ax=plt.subplots(figsize=(6,6))

sns.heatmap(cmnb, annot=True)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train, y_train)

print("Decision Trees Prediction Accuracy:", 100*dt.score(x_test, y_test), "%")
y_true=y_test

y_pred=dt.predict(x_test)

cmdt=confusion_matrix(y_true, y_pred)
f,ax= plt.subplots(figsize=(6,6))

sns.heatmap(cmdt, annot=True)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
from sklearn.ensemble import RandomForestClassifier

score_list=[]

for each in range(1,75):

    rf2=RandomForestClassifier(n_estimators=each, random_state=42)

    rf2.fit(x_train, y_train)

    score_list.append(100*rf2.score(x_test, y_test))

    print("n_estimators=", each, "--> Accuracy:", 100*rf2.score(x_test, y_test), "%")



plt.plot([*range(1,75)], score_list)

plt.xlabel("n_estimators Value")

plt.ylabel("Accuracy %")

plt.show()
optimal_n_estimators_value=score_list.index(max(score_list))+1

rf=RandomForestClassifier(n_estimators=optimal_n_estimators_value, random_state=42)

rf.fit(x_train, y_train)

print("Random Forest Prediction Accuracy:", 100*rf.score(x_test, y_test), "%")
y_true=y_test

y_pred=rf.predict(x_test)

cmrf=confusion_matrix(y_true, y_pred)
f,ax= plt.subplots(figsize=(6,6))

sns.heatmap(cmrf, annot=True)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
print("Logistic Regression Accuracy:", 100*lr.score(x_test, y_test), "%")

print("KNN Prediction Accuracy:", 100*knn.score(x_test, y_test), "%")

print("SVM Prediction Accuracy:", 100*svm.score(x_test, y_test), "%")

print("Naive Bayes Prediction Accuracy:", 100*nb.score(x_test, y_test), "%")

print("Decision Trees Prediction Accuracy:", 100*dt.score(x_test, y_test), "%")

print("Random Forest Prediction Accuracy:", 100*rf.score(x_test, y_test), "%")