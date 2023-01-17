# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np #library used to covert input data to array/metrix

import pandas as pd #library used for data manipulation analysis and cleaning

from matplotlib import pyplot as plt #used to plot the graphs

from sklearn.linear_model import LinearRegression #model we use 
#input data set from kaggle using the path 

#two data set are used one for training the model and another for testing the model

train_dirty = pd.read_csv('../input/random-linear-regression/train.csv')

test_dirty = pd.read_csv('../input/random-linear-regression/test.csv')

#drop the rows having missing values(data cleaning)

train = train_dirty.dropna() 

test = test_dirty.dropna() 
#print the first five rows of the dataset 

print("****taining dataset****")

print(train.head())

print("\n")

print("****testing dataset****")

print(test.head())
#coverting dataset to an array using numpy

X_train = np.array(train.iloc[ : ,:1].values)#x coloumn from the training dataset is assigned to X_train as an array

Y_train = np.array(train.iloc[:, 1].values)#y coloumn from the training dataset is assigned to Y_train as an array

X_test = np.array(test.iloc[:, : 1].values)#x coloumn from the teasting dataset is assigned to X_test as an array

Y_test = np.array(test.iloc[:, 1].values)#y coloumn from the teasting dataset is assigned to Y_test as an array
#plotting the datapoints

plt.scatter(X_train,Y_train, color = "m",marker = "o", s = .5) 
model = LinearRegression()#linear regression model is called

model.fit(X_train, Y_train)#model is trained with X_train and Y_train
Y_pred = model.predict(X_test)#save the predicted output of the model for data X_test
plt.scatter(X_train,Y_train, color = "m",marker = "o", s = 1) #plot the training data points with X_train as x_axis and Y_train as y_axis

plt.plot(X_train, model.predict(X_train), color='green')#plot graph
accuracy = model.score(X_test, Y_test)

print(accuracy)#print the accuracy