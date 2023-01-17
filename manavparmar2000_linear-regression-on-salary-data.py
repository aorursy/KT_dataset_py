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


#including the dataset

data = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")

data.head(5)

#independent variable

x = data.iloc[:, [0]]

#dependent variable

y = data.iloc[:, [1]]
#spliting data into train and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)
x_train
y_train
x_test
y_test
#fitting the train data in linearmodel

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)
#predicting y based on accuracy of trained model

y_pred = reg.predict(x_test)
#predicted y values

y_pred
#original y values

y_test
#linear regression of trained model

import matplotlib.pyplot as plt

plt.scatter(x_train, y_train, color = 'red')

plt.plot(x_train, reg.predict(x_train), color = 'blue')

plt.xlabel('years of experience')

plt.ylabel('Salary')

plt.title('linear regression')

plt.show()
#linear regression of tested model

plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_test, reg.predict(x_test), color = 'blue')

plt.xlabel('years of experience')

plt.ylabel('Salary')

plt.title('linear regression')

plt.show()