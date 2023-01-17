# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")
iris.head()
iris.info()
iris.drop('Id',axis=1,inplace=True)
fig = iris[iris.Species=='Iris-setosa'].plot(kind='Scatter',x='SepalLengthCm',y='SepalWidthCm'

                                             ,color='Orange',label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='Scatter',x='SepalLengthCm',y='SepalWidthCm'

                                           ,color='Blue',label='Versicolor', ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='Scatter',x='SepalLengthCm',y='SepalWidthCm'

                                          ,color='Green',label='Virginica', ax=fig)

fig.set_xlabel('Sepal Length')

fig.set_ylabel('Sepal Width')

fig.set_title('Sepal Length Vs Width Scatter Plot')

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig=iris[iris.Species=='Iris-setosa'].plot(kind='Scatter',x='PetalLengthCm',y='PetalWidthCm',color='Orange',label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='Scatter',x='PetalLengthCm',y='PetalWidthCm',color='Blue',label='Versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='Scatter',x='PetalLengthCm',y='PetalWidthCm',color='Green',label='Virginica',ax=fig)

fig.set_xlabel='Petal Length'

fig.set_ylabel='Petal Width'

fig.set_title='Petal Length Vs Width Scatter Plot'

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
iris.hist(edgecolor='black',linewidth='1.5')

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris,label='Species Vs Petal Length')

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris,label='Species Vs Petal Width')

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris,label='Species Vs Sepal Length')

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris,label='Species Vs Sepal Width')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
iris.shape
plt.figure(figsize=(15,10))

sns.heatmap(iris.corr(),annot=True,color='red')

plt.show()
train,test=train_test_split(iris,test_size=0.3)

print(train.shape)

print(test.shape)
x_train=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_train=train.Species

x_test=test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_test=test.Species
x_train.head(5)
x_test.head(5)
y_train.head(5)
model=svm.SVC()

model.fit(x_train,y_train)

prediction1=model.predict(x_test)

print('The accuracy of SVM is:' ,metrics.accuracy_score(prediction1,y_test))
model=LogisticRegression()

model.fit(x_train,y_train)

prediction2=model.predict(x_test)

print('The accuracy of Logistic Regression is:' ,metrics.accuracy_score(prediction2,y_test))
model=DecisionTreeClassifier()

model.fit(x_train,y_train)

prediction3=model.predict(x_test)

print('The accuracy score of Decision Tree is:' ,metrics.accuracy_score(prediction3,y_test))
model=KNeighborsClassifier()

model.fit(x_train,y_train)

prediction4=model.predict(x_test)

print('The model accuracy score of K Neighbors Classifier is:' ,metrics.accuracy_score(prediction4,y_test))
model=RandomForestClassifier()

model.fit(x_train,y_train)

prediction5=model.predict(x_test)

print('The accuracy of Random Forest Classifier is:' ,metrics.accuracy_score(prediction5,y_test))
df={'Model Name':['Support Vector Machine','Logistic Regression','Decision Tree','K Neighbors',

                 'Random Forest Classifier'],

   'Accuracy':[metrics.accuracy_score(prediction1,y_test),metrics.accuracy_score(prediction2,y_test),

              metrics.accuracy_score(prediction3,y_test),metrics.accuracy_score(prediction4,y_test),

              metrics.accuracy_score(prediction5,y_test)]}

model_accuracy=pd.DataFrame(df,columns=['Model Name','Accuracy'])

print(model_accuracy)
fig=model_accuracy.plot(kind='bar',x='Model Name',y='Accuracy')

fig.set_ylabel='Accuracy'

fig.set_title='Multiple Model Accuracy Graph'

fig=plt.gcf()

fig.set_size_inches(8,6)

plt.show()