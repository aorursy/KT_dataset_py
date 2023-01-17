# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 



# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Using pandas function read_csv to read the csv file

iris = pd.read_csv("../input/iris/Iris.csv")
# Below command display the first 5 rows. Since the number of rows is not defined, the default value is 5 and hence this command displays the first 5 records

iris.head()
# To drop the "Id" column we use the drop function. We also specify the axis. Meaning, if we are dropping a column then axis = 1 and if we are dropping a specific row, then axis=0

# Also, inplace implies that we want the changes to be reflected in the variable "iris" dataframe

iris.drop("Id", axis=1, inplace=True)
# again viewing the data to confirm that we dropped the "Id" column from our dataframe

iris.head()
# Lets take a look at the shape of the data set. This gives us an idea about the number of rows and columns that are present

iris.shape
iris.describe()
iris.info()
sns.FacetGrid(iris, hue="Species", height=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "SepalLengthCm").add_legend()
sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "SepalWidthCm").add_legend()
sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "PetalLengthCm").add_legend()
sns.FacetGrid(iris, hue="Species", height=6).map(sns.kdeplot, "PetalWidthCm").add_legend()
sns.pairplot(iris, hue="Species", height=3)
#importing the required libraries

from sklearn.linear_model import LogisticRegression     #importing Logistic Regression

from sklearn.model_selection import train_test_split   #for splitting the data into train and test sets

from sklearn.neighbors import KNeighborsClassifier    #importing K nearest neighbors algo

from sklearn import svm                                 #importing Support vector machines

from sklearn import metrics                             #for checking the algo performance

from sklearn.tree import DecisionTreeClassifier         #importing decision tree classifier
sns.heatmap(iris.corr(), annot=True)
#Splitting the data into training and test set

train, test = train_test_split(iris, test_size=0.3) #the test_size=0.3 splits the train and test data in the ratio of 70:30



print(train.shape)

print(test.shape)
train_X = train[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

train_y = train["Species"]



test_X = test[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

test_y = test["Species"]

train_X.head()
model = svm.SVC()                     # selecting the algo

model.fit(train_X, train_y)           # we train the algo with the training data and training output

prediction = model.predict(test_X)    # now we pass the test data to the trained algo

acc_score = metrics.accuracy_score(prediction, test_y) # here we check the accuracy of our model. In order to do that,

                                                        # we pass the predicted value by the model and actual value

print("The accuracy of SVM algo is: ", acc_score)
model = LogisticRegression()                     # selecting the algo

model.fit(train_X, train_y)           # we train the algo with the training data and training output

prediction = model.predict(test_X)    # now we pass the test data to the trained algo

acc_score = metrics.accuracy_score(prediction, test_y) # here we check the accuracy of our model. In order to do that,

                                                        # we pass the predicted value by the model and actual value

print("The accuracy of Logistic Regression algo is: ", acc_score)
model = DecisionTreeClassifier()                     # selecting the algo

model.fit(train_X, train_y)           # we train the algo with the training data and training output

prediction = model.predict(test_X)    # now we pass the test data to the trained algo

acc_score = metrics.accuracy_score(prediction, test_y) # here we check the accuracy of our model. In order to do that,

                                                        # we pass the predicted value by the model and actual value

print("The accuracy of Decision Tree classifier algo is: ", acc_score)
model=KNeighborsClassifier(n_neighbors=3)

model.fit(train_X, train_y)           # we train the algo with the training data and training output

prediction = model.predict(test_X)    # now we pass the test data to the trained algo

acc_score = metrics.accuracy_score(prediction, test_y) # here we check the accuracy of our model. In order to do that,

                                                        # we pass the predicted value by the model and actual value

print("The accuracy of KNN algo is: ", acc_score)
petal = iris[["PetalLengthCm", "PetalWidthCm", "Species"]]

sepal = iris[["SepalLengthCm", "SepalWidthCm", "Species"]]
sepal.head()
train_p, test_p = train_test_split(petal, test_size=0.3, random_state=0)

train_x_p = train_p[["PetalLengthCm", "PetalWidthCm"]]

train_y_p = train_p["Species"]

test_x_p = test_p[["PetalLengthCm", "PetalWidthCm"]]

test_y_p = test_p["Species"]



train_s, test_s = train_test_split(sepal, test_size=0.3, random_state=0)

train_x_s = train_s[["SepalLengthCm", "SepalWidthCm"]]

train_y_s = train_s["Species"]

test_x_s = test_s[["SepalLengthCm", "SepalWidthCm"]]

test_y_s = test_s["Species"]

model = svm.SVC()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

accu_score = metrics.accuracy_score(prediction, test_y_p)

print("The accuracy of petals using SVM is: ", accu_score)



model = svm.SVC()

model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

accu_score = metrics.accuracy_score(prediction, test_y_s)

print("The accuracy of sepals using SVM is: ", accu_score)
model = LogisticRegression()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

accu_score = metrics.accuracy_score(prediction, test_y_p)

print("The accuracy of petals using Logistic Regression is: ", accu_score)



model = LogisticRegression()

model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

accu_score = metrics.accuracy_score(prediction, test_y_s)

print("The accuracy of sepals using Logistic Regression is: ", accu_score)
model = DecisionTreeClassifier()

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

accu_score = metrics.accuracy_score(prediction, test_y_p)

print("The accuracy of petals using Decision Tree is: ", accu_score)



model =  DecisionTreeClassifier()

model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

accu_score = metrics.accuracy_score(prediction, test_y_s)

print("The accuracy of sepals using Decision Tree is: ", accu_score)
model = KNeighborsClassifier(n_neighbors=3)

model.fit(train_x_p, train_y_p)

prediction = model.predict(test_x_p)

accu_score = metrics.accuracy_score(prediction, test_y_p)

print("The accuracy of petals using KNN is: ", accu_score)



model = KNeighborsClassifier(n_neighbors=3)

model.fit(train_x_s, train_y_s)

prediction = model.predict(test_x_s)

accu_score = metrics.accuracy_score(prediction, test_y_s)

print("The accuracy of sepals using KNN is: ", accu_score)