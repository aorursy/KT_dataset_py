# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing all libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





#importing the dataset

dataset=pd.read_csv("../input/train.csv")

dataset1=pd.read_csv("../input/test.csv")
#how thetraining set looks like



dataset
#how does the test set looks like

dataset1

dataset.isnull().sum()
dataset1.isnull().sum()
dataset.shape
dataset.dropna().shape
dataset.isnull().sum()
dataset=dataset.dropna()

dataset.isnull().sum()


X_train = dataset.iloc[:,:-1].values

y_train = dataset.iloc[:,1].values

X_test = dataset1.iloc[:,:-1].values

y_test = dataset1.iloc[:,1].values
#Applying regeression to the given datatset



from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

#visualisation for training set

plt.scatter(X_train,y_train,color="red")

plt.plot(X_train,regressor.predict(X_train),color="blue")

plt.title('Linear Regression(training Set)')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()
#visualisation for the test values

plt.scatter(X_test,y_test,color="red")

plt.plot(X_train,regressor.predict(X_train),color="blue")

plt.title('Linear Regression(Test Set)')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()