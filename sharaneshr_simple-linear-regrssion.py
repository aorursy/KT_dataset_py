# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test_dirty = pd.read_csv("../input/random-linear-regression/test.csv")

train_dirty = pd.read_csv("../input/random-linear-regression/train.csv")
train_dirty.size,test_dirty.size
test = test_dirty.dropna()#dropping if there are any undefined terms in the datasets

train = train_dirty.dropna()
x_train= train.as_matrix(['x']) #copying the column with x as the attribute to x_train and //ly for y_train

y_train= train.as_matrix(['y'])

x_train.size,y_train.size
x_test= test.as_matrix(['x'])

y_test= test.as_matrix(['y'])

x_test.size,y_test.size
from sklearn.linear_model import LinearRegression #importing the linear regression model from sklearn

lr=LinearRegression()

lr.fit(x_train,y_train) #fitting he model based on the training dataset

y_pred=lr.predict(x_test) #predicting the output for the values of y

y_act_pred=lr.predict(x_train)

#lr.coef_,lr.intercept_
from sklearn.metrics import r2_score

r2_test=r2_score(y_test,y_pred)

r2_train=r2_score(y_train,y_act_pred)

print("The training R2_Score is: ",r2_train,"\n")

print("The testing R2_Score is: ",r2_test,"\n")
plt.subplot(1,2,1)

plt.title("Actual value of Y_test")

plt.scatter(x_test,y_test)



plt.subplot(1,2,2)

plt.title("Predicted value of Y_test")

plt.scatter(x_test,y_pred)