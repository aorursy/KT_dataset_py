# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
iris=pd.read_csv("../input/iris/Iris.csv")
iris.head(5) #shows the first 5 rows from the dataset
iris.drop('Id', axis=1 , inplace=True) #dropping the Id column as it is redundant. axis=1 means we are removing the column

#inplace=True means that the changes will be reflected in our dataframe.
iris.head(2)
fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', label='Setosa', color='orange')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', label='Versicolor', color='blue', ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', label='Virginica', color='green', ax=fig)

fig.set_xlabel('Sepal Length in cm')

fig.set_ylabel('Sepal Width in cm')

fig.set_title('SepalLength vs SepalWidth')

fig=plt.gcf()

fig.show()
fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', label='Setosa', color='orange')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', label='Versicolor', color='blue', ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', label='Virginica', color='green', ax=fig)

fig.set_xlabel('PetalLengthinCm')

fig.set_ylabel('PetalLengthinCm')

fig.set_title('PetalLength vs PetalWidth')

fig=plt.gcf()

fig.show()
iris.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

fig.show()
#importing all the necessary packages to use the various classification algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier 

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier
iris.shape
sns.heatmap(iris.corr(), annot=True)
train, test=train_test_split(iris, test_size=0.3)
print (train.shape)

print (test.shape)
train_x=train[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

train_y=train.Species

test_x=test[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

test_y=test.Species
from sklearn import metrics
model=svm.SVC()

model.fit(train_x, train_y)

prediction=model.predict(test_x)

print("The accuarcy of the SVM model is",metrics.accuracy_score(prediction, test_y))

model=LogisticRegression()

model.fit(train_x, train_y)

model.predict(test_x)

print("The accuracy of the Logistic Regression Model is ",metrics.accuracy_score(prediction, test_y))
model=DecisionTreeClassifier()

model.fit(train_x, train_y)

model.predict(test_x)

print ("The accuracy of the model is ",metrics.accuracy_score(prediction, test_y))
model=KNeighborsClassifier(n_neighbors=3)

model.fit(train_x,train_y)

model.predict(test_x)

print("The accuracy of this model is ",metrics.accuracy_score(prediction, test_y))
