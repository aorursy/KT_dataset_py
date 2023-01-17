# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.head()
data.tail()
data.info()
print(data.loc[:,"target"].value_counts())
sns.countplot(data["target"])
data.isnull().any()
x_data=data.drop(["target"],axis=1)
y=data["target"].values
x_data.head()
y
# Normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# Train,test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()
lr.fit(x,y) # Modelimizi eğitiriz.
print("Train Accuracy: {}".format(lr.score(x_train,y_train)))
print("Test Accuracy: {}".format(lr.score(x_test,y_test)))
list_1=[]
list_2=[]

list_1.append("LogisticRegression")
list_2.append(lr.score(x_test,y_test)*100)
# KNN

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train,y_train)
print("Train Accuracy: {}".format(knn.score(x_train,y_train)))
print("Test Accuracy: {}".format(knn.score(x_test,y_test)))
# Yukarıda k=6 için accuracy değeri bulduk. Peki diğer k değerlerinde accuracy ne çıkacak ? Daha iyi bir accuracy değeri bulabiliriz.

score_test=[]
score_train=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    print("Test: {}".format(knn2.score(x_test,y_test)))
    score_test.append(knn2.score(x_test,y_test))
    score_train.append(knn2.score(x_train,y_train))
    
plt.plot(range(1,15),score_train,color="blue",label="train")
plt.plot(range(1,15),score_test,color="purple",label="test")
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.legend()
plt.show()
list_1.append("KNN")
list_2.append(knn.score(x_test,y_test)*100)
# SVM

from sklearn.svm import SVC

svm=SVC()
svm.fit(x_train,y_train)
print("Train Accuracy: {}".format(svm.score(x_train,y_train)))
print("Test Accuracy: {}".format(svm.score(x_test,y_test)))
list_1.append("SVM")
list_2.append(svm.score(x_test,y_test)*100)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
nb.fit(x_train,y_train)
print("Train Accuracy: {}".format(nb.score(x_train,y_train)))
print("Test Accuracy: {}".format(nb.score(x_test,y_test)))
list_1.append("NaiveBayes")
list_2.append(nb.score(x_test,y_test)*100)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Train Accuracy: {}".format(dt.score(x_train,y_train)))
print("Test Accuracy: {}".format(dt.score(x_test,y_test)))
list_1.append("DecisionTree")
list_2.append(dt.score(x_test,y_test)*100)
# Random Forest 

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100) # n_estimators ile kaç tane ağaç kullanacağımızı belirtiriz.
rf.fit(x_train,y_train)
print("Train Accuracy: {}".format(rf.score(x_train,y_train)))
print("Test Accuracy: {}".format(rf.score(x_test,y_test)))
list_1.append("RandomForest")
list_2.append(rf.score(x_test,y_test)*100)
list_1
list_2
plt.figure(figsize=(15,8))
sns.barplot(x=list_1,y=list_2)
plt.show()
