# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls","../input"]).decode("utf8"))



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/iris/Iris.csv")
df.head(10)
df.info()
df.columns
df['Species'].value_counts()
df['SepalLengthCm'].describe()
dfx=df.drop('Id',axis=1)
dfx.head(2)
fig = dfx[dfx.Species=='Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm',color='orange',label='Setosa')

dfx[dfx.Species=='Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm',color='green',label='virginica',ax=fig)

dfx[dfx.Species=='Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm',color='red',label='versicolor',ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length Vs Width")



fig=dfx[dfx.Species=="Iris-setosa"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange',label='setosa')

dfx[dfx.Species=="Iris-virginica"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green',label='virginica',ax=fig)

dfx[dfx.Species=="Iris-versicolor"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='red',label='versicolor',ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title("Petal Length Vs Width")

plt.show()
dfx.hist(edgecolor='black',linewidth=1)

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=dfx)

plt.subplot(2,2,2)

sns.violinplot(x="Species",y='PetalWidthCm',data=dfx)

plt.subplot(2,2,3)

sns.violinplot(x="Species",y="SepalWidthCm", data=dfx)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y="SepalLengthCm", data=dfx)
from sklearn.linear_model import LogisticRegression #for logistic regression algo

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier #for k nearest neighbors

from sklearn import svm #support vector machine algo

from sklearn import metrics # for checking model accuracy

from sklearn.tree import DecisionTreeClassifier # for using decicion tree algo
plt.figure(figsize=(5,5))

sns.heatmap(dfx.corr(),annot=True,cmap='cubehelix_r')

plt.show()

train, test = train_test_split(dfx,test_size=(0.3))

print(train.shape)

print(test.shape)
train_X=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_Y=train.Species

test_X=test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_Y=test.Species

model=svm.SVC() # define the model

model.fit(train_X,train_Y) #pass the tarining data into the model

prediction=model.predict(test_X)

print('the accuracy score of the svm is:',metrics.accuracy_score(prediction,test_Y))

model=LogisticRegression()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print("Accuracy Score:",metrics.accuracy_score(prediction,test_Y))

model=DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print("Accuracy Score:",metrics.accuracy_score(prediction,test_Y))
train,test = train_test_split(dfx,test_size=(0.3)) #divide the dataframe into training and testing data
train_X = train[['SepalLengthCm','SepalWidthCm']] #create the feature data 

train_Y = train.Species #create the target dataframe

test_X = train[['SepalLengthCm','SepalWidthCm']]

test_Y = train.Species

model = svm.SVC() #define the model

model.fit(train_X,train_Y) #pass the training data to the algo

prediction = model.predict(test_X)

print("The Accuracy Score is :",metrics.accuracy_score(prediction,test_Y))
train_X_P = train[['PetalLengthCm','SepalWidthCm']]

train_Y_P = train.Species

test_X_P = train[['PetalLengthCm','SepalWidthCm']]

test_Y_P = train.Species

model = svm.SVC()

model.fit(train_X_P,train_Y_P)

prediction = model.predict(test_X_P)

print("The Accuracy Score is :", metrics.accuracy_score(prediction,test_Y_P))