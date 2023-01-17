# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



pd.plotting.register_matplotlib_converters()

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Read the csv file using the read_csv method in pandas library

iris_df = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")
# Show the top rows in the dataset

iris_df.head()
# Show the lower rows in the dataset

iris_df.tail()
# show a matrix scatter plot that allows to see different relationships in the dataset

sns.pairplot(data = iris_df, hue = "species", corner = True, markers = "+")
# import sklearn module

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from sklearn.metrics import accuracy_score
# Create a label encoder to transform String y labels to integer labels

enc = LabelEncoder()

# Perform the actual work

iris_df["species"] = enc.fit_transform(iris_df["species"])

# Print the top rows

iris_df.head()
# divide the dataset in train and test datasets

train, test = train_test_split(iris_df, test_size=0.15, shuffle = True)



print("shape of train dataset ", train.shape)

print("shape of test dataset", test.shape)
# transform the datasets into numpy arrays

train_X = train[["sepal_length", "petal_width"]].to_numpy()

tmp = train[["species"]].to_numpy()

train_y = np.reshape(tmp, tmp.shape[0])



test_X = test[["sepal_length", "petal_width"]].to_numpy()

tmp = test[["species"]].to_numpy()

test_y = np.reshape(tmp, tmp.shape[0])



print("shape of train_X dataset ", train_X.shape)

print("shape of train_y dataset ", train_y.shape)

print("shape of test_X dataset", test_X.shape)

print("shape of test_y dataset", test_y.shape)
# create a logistic regression and fit the model

svc = LogisticRegression(max_iter = 250)

svc.fit(train_X, train_y)
# predict the test dataset to get predictions and print the accuracy

pred_y = svc.predict(test_X)



print("The mean accuracy on the given test data and labels is: ", svc.score(test_X, test_y))

print("The accuracy classification score", accuracy_score(test_y, pred_y))
# create a line function that divides the two label clases

p_x = np.arange(-20, 20)



# define the line to be printed by the general eq of the line Ax + By + C = 0

def line(w0, w1, b, x):

    return -(w0 * x + b) / w1



lines = []



for W, b in zip(svc.coef_, svc.intercept_):

    p_y = [line(W[0], W[1], b, x) for x in p_x]

    lines.append(p_y)
# get the limits of the iris dataset

xmin, xmax = iris_df["sepal_length"].min(), iris_df["sepal_length"].max()

ymin, ymax = iris_df["petal_width"].min(), iris_df["petal_width"].max()
# create a scatter plot with the line function

figure = plt.figure(figsize=(26, 9))

ax = plt.subplot(1, 1, 1)



# set the visible limits on the graphic

plt.ylim((ymin - .7, ymax + .7))

plt.xlim((xmin - .2, xmax + .2))



# plot the points

ax.scatter(train_X[:,0], train_X[:,1], c=train_y)

ax.scatter(test_X[:,0],  test_X[:,1], c=test_y)



# plot the boundary lines

for y in lines:

    plt.plot(p_x, y)
plot_confusion_matrix(svc, test_X, test_y, cmap=plt.cm.Blues)
# import the packages

from sklearn.svm import LinearSVC
svm = LinearSVC(max_iter = 3400)

svm.fit(train_X, train_y)
pred_y = svm.predict(test_X)



print("The mean accuracy on the given test data and labels is: ", svm.score(test_X, test_y))

print("The accuracy classification score", accuracy_score(test_y, pred_y))
# create a line function that divides the two label clases

p_x = np.arange(-20, 20)



# define the line to be printed by the general eq of the line Ax + By + C = 0

def line(w0, w1, b, x):

    return -(w0 * x + b) / w1



lines = []



for W, b in zip(svm.coef_, svm.intercept_):

    p_y = [line(W[0], W[1], b, x) for x in p_x]

    lines.append(p_y)
# create a scatter plot with the line function

figure = plt.figure(figsize=(26, 9))

ax = plt.subplot(1, 1, 1)



# set the visible limits on the graphic

plt.ylim((ymin - .7, ymax + .7))

plt.xlim((xmin - .2, xmax + .2))



# plot the points

ax.scatter(train_X[:,0], train_X[:,1], c=train_y)

ax.scatter(test_X[:,0],  test_X[:,1], c=test_y)



# plot the boundary lines

for y in lines:

    plt.plot(p_x, y)
plot_confusion_matrix(svm, test_X, test_y, cmap=plt.cm.Blues)