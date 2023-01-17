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
#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue Jan 28 13:20:06 2020



@author: root

"""





#importing libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#importing Dataset

dataset = pd.read_csv("../input/heights-and-weights/data.csv")

X = dataset.iloc[:,:-1] #height

y = dataset.iloc[:, 1] #weight





#splitting dataset into trainin set and test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)



#fitting the regression model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)



#predicting the test set result

y_pred = regressor.predict(X_test)



#visualization of training set results

plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color="blue")

plt.title('height vs weight(training  set)')

plt.xlabel('heigt')

plt.ylabel('weight')

plt.show()



#visualization of test set results

plt.scatter(X_test,y_test,color='red')

plt.plot(X_train,regressor.predict(X_train),color="blue")

plt.title('height vs weight(test set')

plt.xlabel('height')

plt.ylabel('weight')

plt.show()

# Let's see score of our model.

regressor.score(X_test,y_test)
