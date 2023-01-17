# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk # machine learning library

import matplotlib.pyplot as plt # plot data for visualization

import seaborn as sns # statistical data visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# Any results you write to the current directory are saved as output.

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv('../input/iris/Iris.csv') # Import dataset



print(data.head()) # Check data's first rows

data.info() # Check if there are any null values



data.drop('Id',axis=1,inplace=True) # Remove Id column since it is not useful, axis=1 indicates that it is a column drop

                                    # inplace=True means the change should be applied to the data dataframe directly



print(data.head()) # Check data's first rows
# Let's plot the data to get the relationship between sepal width and sepal length



fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='#7F00FF',label='Setosa') # scatter plot, setosa,color violet

data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='#808080',label='versicolor',ax=fig) # scatter plot, versicolor,color gray

data[data.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='#8B0000',label='virginica',ax=fig) # scatter plot, virginica, color red

fig.set_xlabel("Sepal Length") # plot x axis label

fig.set_ylabel("Sepal Width") # plot y axis label

fig.set_title("Sepal Length") # plot title

fig = plt.gcf() # get current figure

fig.set_size_inches(10,6) # plot figure size

plt.show()
# Let's plot the data to get the relationship between petal width and petal length



fig = data[data.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='#7F00FF',label='Setosa') # scatter plot, setosa,color violet

data[data.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='#808080',label='versicolor',ax=fig) # scatter plot, versicolor,color gray

data[data.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='#8B0000',label='virginica',ax=fig) # scatter plot, virginica, color red

fig.set_xlabel("Petal Length") # plot x axis label

fig.set_ylabel("Petal Width") # plot y axis label

fig.set_title("Petal Length") # plot title

fig = plt.gcf() # get current figure

fig.set_size_inches(10,6) # plot figure size

plt.show()
# Let's see how the length and width are distributed



data.hist(edgecolor='black',linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
from sklearn.linear_model import LogisticRegression # for logistic regression

from sklearn.model_selection import train_test_split # to split the dataset into trainset and testset

from sklearn.neighbors import KNeighborsClassifier # for k nearest neighbours

from sklearn import svm # for support vector machine(SVM) algorithm

from sklearn import metrics # for checking the  model accuracy

from sklearn.tree import DecisionTreeClassifier # for decision tree
data.shape # Get dataset's shape



plt.figure(figsize=(7,4))

sns.heatmap(data.corr(),annot=True,cmap='cubehelix_r') # draws heatmap with input as the correlation matrix calculated by (data.corr())

plt.show()
# Now let's split the data into training and test datasets



train,test = train_test_split(data, test_size=0.3) # train data = 70% and test data = 30%



# Let's see how the training and test shapes are looking

print(train.shape)

print(test.shape)
# Let's create variables for the input and output variables both for the training and testing datasets



train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # training data features

train_y = train.Species # output of training data

test_X = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] # test data features

test_y = test.Species # output value of test data



print(train_X.head())

print(train_y.head())
# Let's start by training with the Support Vector Machine (SVM) algorithm



model = svm.SVC() # select the algorithm

model.fit(train_X,train_y) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X) # now we pass the testing data to the trained algorithm

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output
# Let's now train using the LogisticRegression model



model = LogisticRegression()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(prediction,test_y))
# Let's now train using the KNeighbors Classifier model



model = KNeighborsClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the KNeighbors Classifier is:',metrics.accuracy_score(prediction,test_y))
# Let's now train using the Decision Tree Classifier model



model = DecisionTreeClassifier()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

print('The accuracy of the Decision Tree Classifier is:',metrics.accuracy_score(prediction,test_y))
# Now let's check the accuracy using the KNeighbours Classifier for n neighbours



a_index=list(range(1,11))

a=pd.Series()

x_axis=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X,train_y)

    prediction=model.predict(test_X)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))

plt.plot(a_index, a)

plt.xticks(x_axis)
# Now let's split the data into training and test datasets using Petal and sepal as features



petal = data[['PetalLengthCm','PetalWidthCm','Species']]

sepal = data[['SepalLengthCm','SepalWidthCm','Species']]



train_p,test_p = train_test_split(petal, test_size=0.3,random_state=0) # train data = 70% and test data = 30% for Petals

train_X_p = train_p[['PetalLengthCm','PetalWidthCm']] # training data features

train_y_p = train_p.Species # output of training data

test_X_p = test_p[['PetalLengthCm','PetalWidthCm']] # test data features

test_y_p = test_p.Species # output value of test data



train_s,test_s = train_test_split(sepal, test_size=0.3,random_state=0) # train data = 70% and test data = 30% for Sepals

train_X_s = train_s[['SepalLengthCm','SepalWidthCm']] # training data features

train_y_s = train_s.Species # output of training data

test_X_s = test_s[['SepalLengthCm','SepalWidthCm']] # test data features

test_y_s = test_s.Species # output value of test data



# Let's see how the training and test shapes are looking

print(train_p.shape)

print(test_p.shape)

print(train_s.shape)

print(test_s.shape)
# Let's start by training with the Support Vector Machine (SVM) algorithm



model = svm.SVC() # select the algorithm

model.fit(train_X_p,train_y_p) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_p) # now we pass the testing data to the trained algorithm

print('The accuracy of the SVM for Petals is:',metrics.accuracy_score(prediction,test_y_p)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output

    

model = svm.SVC() # select the algorithm

model.fit(train_X_s,train_y_s) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_s) # now we pass the testing data to the trained algorithm

print('The accuracy of the SVM Sepals is:',metrics.accuracy_score(prediction,test_y_s)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output
# Let's start by training with the Logistic Regression algorithm



model = LogisticRegression() # select the algorithm

model.fit(train_X_p,train_y_p) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_p) # now we pass the testing data to the trained algorithm

print('The accuracy of the Logistic Regression for Petals is:',metrics.accuracy_score(prediction,test_y_p)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output

    

model = LogisticRegression() # select the algorithm

model.fit(train_X_s,train_y_s) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_s) # now we pass the testing data to the trained algorithm

print('The accuracy of the Logistic Regression Sepals is:',metrics.accuracy_score(prediction,test_y_s)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output
# Let's start by training with the KNeighbors Classifier algorithm



model = KNeighborsClassifier() # select the algorithm

model.fit(train_X_p,train_y_p) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_p) # now we pass the testing data to the trained algorithm

print('The accuracy of the KNeighbors Classifier for Petals is:',metrics.accuracy_score(prediction,test_y_p)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output

    

model = KNeighborsClassifier() # select the algorithm

model.fit(train_X_s,train_y_s) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_s) # now we pass the testing data to the trained algorithm

print('The accuracy of the KNeighbors Classifier Sepals is:',metrics.accuracy_score(prediction,test_y_s)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output
# Now let's check the accuracy using the KNeighbours Classifier for n neighbours



a_index=list(range(1,11))

a=pd.Series()

x_axis=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X_p,train_y_p)

    prediction=model.predict(test_X_p)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y_p)))

plt.plot(a_index, a)

plt.xticks(x_axis)



a_index=list(range(1,11))

a=pd.Series()

x_axis=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,11)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(train_X_s,train_y_s)

    prediction=model.predict(test_X_s)

    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y_s)))

plt.plot(a_index, a)

plt.xticks(x_axis)
# Let's start by training with the Decision Tree Classifier algorithm



model = DecisionTreeClassifier() # select the algorithm

model.fit(train_X_p,train_y_p) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_p) # now we pass the testing data to the trained algorithm

print('The accuracy of the Decision Tree Classifier for Petals is:',metrics.accuracy_score(prediction,test_y_p)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output

    

model = DecisionTreeClassifier() # select the algorithm

model.fit(train_X_s,train_y_s) # we train the algorithm with the training data and the training output

prediction=model.predict(test_X_s) # now we pass the testing data to the trained algorithm

print('The accuracy of the Decision Tree Classifier Sepals is:',metrics.accuracy_score(prediction,test_y_s)) # now we check the accuracy of the algorithm. 

                                                                               # we pass the predicted output by the model and the actual output