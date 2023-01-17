# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")
iris.head(5)
iris.head(5)
iris.info()
fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
iris.drop('Id',axis=1,inplace=True) 
iris.head(5)
plt.figure(figsize=(7,4)) 

sns.heatmap(iris.corr(),annot=True,cmap='cubehelix_r')

plt.show()
from sklearn.model_selection import train_test_split #to split the dataset for training and testing

train, test = train_test_split(iris, test_size = 0.3)

print(train.shape)

print(test.shape)
train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y =train.Species

test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_y =test.Species
# importing alll the necessary packages to use the various classification algorithms

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn import metrics #for checking the model accuracy
model=KNeighborsClassifier(n_neighbors=3)

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))
#Change data series to array 

test_y_array = pd.Series(list(test_y)).values

test_y_array
prediction
# y_aja = pd.Categorical(test_y_array)

# y_aja = y_aja.categories.tolist()

# y_aja
# x = [i+1 for i in range (len(test_y_array))]

# x
df1 = {'No': [i+1 for i in range (len(test_y_array))] , 'Species':test_y_array}

df_true = pd.DataFrame(df1)

df2 = {'No': [i+1 for i in range (len(prediction))] , 'Species':prediction}

df_pred = pd.DataFrame(df2)
fig, ax = plt.subplots(figsize=(16,4))

ax.scatter(df_true.No, df_true.Species, color='red', label="True")

ax.scatter(df_pred.No, df_pred.Species, color='blue', label="Prediction")

ax.legend()

ax.set_title('True labels vs Predictions')

plt.show()
#KNN's accuracy (neighbors in range 1-10)

a_index=list(range(1,11))

a=pd.Series()

x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))

plt.plot(a_index, a)

plt.xticks(x)