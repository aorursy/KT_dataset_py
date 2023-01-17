# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Iris.csv")
df.shape
df.count()
fig = plt.figure(figsize = (18,6))
plt.subplot2grid((2,3),(0,0))
plt.scatter(df.SepalLengthCm, df.SepalWidthCm, alpha= 0.5)
plt.title("Sepal Length vs Sepal Width")

df.drop('Id', axis = 1, inplace = True)
df.head()
fig = df[df.Species == 'Iris-setosa'].plot(kind= 'scatter', x='SepalLengthCm', y='SepalWidthCm', color = 'orange', label='Setosa') 
df[df.Species == 'Iris-virginica'].plot(kind= 'scatter', x='SepalLengthCm', y='SepalWidthCm', color = 'blue', label='Virginica', ax=fig) 
df[df.Species == 'Iris-versicolor'].plot(kind= 'scatter', x='SepalLengthCm', y='SepalWidthCm', color = 'green', label='Versicolor', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()
fig = df[df.Species == 'Iris-setosa'].plot(kind= 'scatter', x='PetalLengthCm', y='PetalWidthCm', color = 'orange', label='Setosa') 
df[df.Species == 'Iris-virginica'].plot(kind= 'scatter', x='PetalLengthCm', y='PetalWidthCm', color = 'blue', label='Virginica', ax=fig) 
df[df.Species == 'Iris-versicolor'].plot(kind= 'scatter', x='PetalLengthCm', y='PetalWidthCm', color = 'green', label='Versicolor', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length VS Width")
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True,cmap="cubehelix_r")
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
train, test = train_test_split(df, test_size = 0.4)
print(train.shape)
print(test.shape)
trainx = train [['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
trainy = train.Species
testx = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
testy = test.Species
trainx.head(2)
testx.head(2)
trainy.head()
model = svm.SVC()
model.fit(trainx,trainy)
prediction = model.predict(testx)
print('accuracy for SVM is:',metrics.accuracy_score(prediction,testy))
model = DecisionTreeClassifier()
model.fit(trainx,trainy)
prediction = model.predict(testx)
print('accuracy for Decision Tree Classifier is:',metrics.accuracy_score(prediction,testy))
model = LogisticRegression()
model.fit(trainx,trainy)
prediction = model.predict(testx)
print('accuracy for Logistic Regression is:',metrics.accuracy_score(prediction,testy))
model = KNeighborsClassifier(n_neighbors = 3)
model.fit(trainx,trainy)
prediction = model.predict(testx)
print('accuracy for KNN is:',metrics.accuracy_score(prediction,testy))
a_index=list(range(1,11))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(trainx,trainy)
    prediction=model.predict(testx)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,testy)))
plt.plot(a_index, a)
plt.xticks(x)
