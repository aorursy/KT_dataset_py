import string # library used to deal with some text data

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library

import matplotlib.pyplot as plt # plotting library

df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

display(df.head(10))
df.isnull().sum()
df.info()
df.describe()
fig, ax = plt.subplots(figsize=(10,10))         # Samplefigsize in inches

sns.heatmap(df.corr(), annot=True, linewidths=.5, ax=ax)
df["quality"] = pd.cut(df["quality"],bins=[0,6.5,10],labels=["Good","Bad"])

df.head()
df.quality.value_counts().plot(kind='pie')
from sklearn.cluster import KMeans

from sklearn import preprocessing,svm,neighbors

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

df=df[["quality","alcohol","sulphates","citric acid","fixed acidity","residual sugar"]]

X=df.drop(["quality"],1)

y=df["quality"]


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=1)

lr=LogisticRegression()

knn=neighbors.KNeighborsClassifier(n_neighbors = 5) 

svm0=svm.SVC(random_state = 1)

nb=GaussianNB()

dtc = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators = 3000, random_state = 1)



lr.fit(X_train,y_train)

knn.fit(X_train,y_train)

svm0.fit(X_train,y_train)

nb.fit(X_train,y_train)

dtc.fit(X_train,y_train)

rf.fit(X_train,y_train)



accuracy={}

accuracy["Logisticregression"]=lr.score(X_test,y_test)*100

accuracy["Knn"]=knn.score(X_test,y_test)*100

accuracy["SVM"]=svm0.score(X_test,y_test)*100

accuracy["Naive Bayes"]=nb.score(X_test,y_test)*100

accuracy["Random Forest"]=rf.score(X_test,y_test)*100

print(accuracy)
p=sns.barplot(list(accuracy.keys()),list(accuracy.values()),palette="bright")

_=plt.setp(p.get_xticklabels(),rotation=90)

plt.show()