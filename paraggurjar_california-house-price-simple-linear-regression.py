# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
import sklearn.linear_model as slm
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split ## Higher Version 0.19.1
from sklearn.cross_validation import train_test_split ## Higher Version 0.16.1

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#data = pd.read_csv('housing.csv')
data = pd.read_csv('../input/housing.csv')
data.head()
data_refine = data.drop('ocean_proximity', axis = 1)
data_refine.info()
data_refine = data_refine.dropna(axis = 0)
data_refine.info()
X = data_refine.drop('median_house_value', axis = 1)
y = data_refine['median_house_value']
X.info()
print(y.shape)
plt.scatter(X['total_bedrooms'], y)
plt.xlabel('total_bedrooms')
plt.ylabel('House Price')
X.head()
color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')
X['total_bedrooms'].plot.box(color=color)
LR = slm.LinearRegression()
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.25)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
LR.fit(X_train, Y_train)
predict = LR.predict(X_test)
print('Predicted Value :',predict[0])
print('Actual Value :',Y_test.values[0])
LR.score (X_test, Y_test)
gr = pd.DataFrame({'Predicted':predict,'Actual':Y_test})
gr = gr.reset_index()
gr = gr.drop(['index'],axis=1)
plt.plot(gr[:1000])
plt.legend(['Actual','Predicted'])
#gr.plot.bar();